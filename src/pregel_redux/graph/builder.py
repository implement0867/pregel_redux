from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any, Concatenate, Literal, Never, Protocol, assert_never, get_args, overload, runtime_checkable

from pydantic import TypeAdapter

from pregel_redux.channels import build_channels, merge_channels
from pregel_redux.checkpoints import Checkpointer
from pregel_redux.commands import Endpoint, read_declared_targets
from pregel_redux.configs import RunTarget
from pregel_redux.dependencies import CallPlan, NodeValue, Scope, Supplied, plan_call
from pregel_redux.graph.compiled import Breakpoints, CompiledGraph, CompiledNode
from pregel_redux.graph.persistent import PersistentGraph
from pregel_redux.messages import MessageSchema
from pregel_redux.pregel import BranchKey, JoinKey
from pregel_redux.shutdown import ShutdownSignal
from pregel_redux.streams import StreamWriter
from pregel_redux.task_refs import StartRef

type Node[InputT] = Callable[Concatenate[InputT, ...], object]


@runtime_checkable
class GraphBuilder(Protocol):
    @property
    def input_schema(self) -> type[object]: ...
    @property
    def dependencies(self) -> frozenset[Callable[..., object]]: ...
    def compile_graph(
        self,
        dependency_overrides: Mapping[Callable[..., object], Callable[..., object]] | None,
        supplied: Mapping[Supplied, Scope],
    ) -> CompiledGraph[object, Never, object]: ...


@dataclass(frozen=True)
class Edge:
    source: str | Literal[Endpoint.start]
    target: str | Literal[Endpoint.end]


@dataclass(frozen=True)
class Join:
    sources: frozenset[str]
    target: str | Literal[Endpoint.end]


class StateGraph[StateT, InputT, OutputT]:
    """Builds a graph of nodes that read and write one shared state.

    Each schema is a `TypedDict`, a dataclass or a Pydantic model.

    Parameters
    ----------
    input_schema
        The fields that the run input writes.
        The default is `state_schema`.
    output_schema
        The fields of the run output.
        The default is `state_schema`.
    """

    @overload
    def __init__[S](self: "StateGraph[S, S, S]", state_schema: type[S]) -> None: ...
    @overload
    def __init__[S, I](self: "StateGraph[S, I, S]", state_schema: type[S], *, input_schema: type[I]) -> None: ...
    @overload
    def __init__[S, O](self: "StateGraph[S, S, O]", state_schema: type[S], *, output_schema: type[O]) -> None: ...
    @overload
    def __init__[S, I, O](
        self: "StateGraph[S, I, O]", state_schema: type[S], *, input_schema: type[I], output_schema: type[O]
    ) -> None: ...
    def __init__(self, state_schema: Any, *, input_schema: Any | None = None, output_schema: Any | None = None) -> None:  # pyright: ignore[reportExplicitAny]
        self.state_schema: type[StateT] = state_schema
        self.input_schema: type[InputT] = state_schema if input_schema is None else input_schema
        self.output_schema: type[OutputT] = state_schema if output_schema is None else output_schema
        self.nodes: dict[str, Callable[..., object]] = {}
        self.node_input_schemas: dict[str, type[object]] = {}
        self.child_graphs: dict[str, GraphBuilder] = {}
        self.edges: set[Edge] = set()
        self.joins: set[Join] = set()
        self.deferred: set[str] = set()

    @overload
    def add_node(self, name: str, node: Node[StateT], *, defer: bool = False) -> None: ...
    @overload
    def add_node[NodeInputT](
        self, name: str, node: Node[NodeInputT], *, input_schema: type[NodeInputT], defer: bool = False
    ) -> None: ...
    @overload
    def add_node[ChildStateT, ChildInputT, ChildOutputT](
        self, name: str, node: "StateGraph[ChildStateT, ChildInputT, ChildOutputT]", *, defer: bool = False
    ) -> None: ...
    def add_node(
        self,
        name: str,
        node: Callable[..., object] | GraphBuilder,
        *,
        input_schema: type[object] | None = None,
        defer: bool = False,
    ) -> None:
        """Adds a node function or a subgraph under `name`.

        A node function receives the state as its first parameter.
        Each other parameter requests an injected value.
        The node function returns a state update or a `Command`.

        Parameters
        ----------
        input_schema
            The type that the node function receives in place of the state.
        defer
            A deferred node runs after each other pending node finishes.

        Examples
        --------
        >>> def summarize(
        ...     state: State, client: Annotated[Client, Depends(open_client)], write: StreamWriter
        ... ) -> Command[Literal["review"]]:
        ...     write("summarizing")
        ...     return Command(update={"summary": client.summarize(state["text"])}, goto="review")
        >>> graph.add_node("summarize", summarize)
        """
        if name in self.nodes or name in self.child_graphs:
            raise ValueError(f"node {name!r} is already in this graph")
        if name == StartRef().name:
            raise ValueError(f"node name {name!r} is the name of the start task in a state snapshot")
        reserved = sorted(set(name) & {"|", ":"})
        if reserved:
            raise ValueError(f"node name {name!r} has {reserved!r}, which separate the parts of a checkpoint namespace")
        if defer:
            self.deferred.add(name)
        match node:
            case GraphBuilder():
                self.child_graphs[name] = node
            case _:
                self.nodes[name] = node
                if input_schema is not None:
                    self.node_input_schemas[name] = input_schema

    def add_edge(
        self,
        source: str | Literal[Endpoint.start] | tuple[str, str, *tuple[str, ...]],
        target: str | Literal[Endpoint.end],
    ) -> None:
        """Runs `target` in the step after `source`.

        A tuple of sources makes `target` wait until each source finishes.
        """
        match source:
            case tuple():
                self.joins.add(Join(frozenset(source), target))
            case _:
                self.edges.add(Edge(source, target))

    @property
    def dependencies(self) -> frozenset[Callable[..., object]]:
        plans = [plan_call(node, {}, list_supplied(persistent=True)) for node in self.nodes.values()]
        return frozenset(
            item.request.dependency for plan in plans for item in (*plan.run_providers, *plan.node_providers)
        ).union(*(child.dependencies for child in self.child_graphs.values()))

    @overload
    def compile[ProviderT: Callable[..., object]](
        self, dependency_overrides: Mapping[ProviderT, Callable[..., object]] | None = None
    ) -> CompiledGraph[StateT, InputT, OutputT]: ...
    @overload
    def compile[ProviderT: Callable[..., object]](
        self,
        dependency_overrides: Mapping[ProviderT, Callable[..., object]] | None = None,
        *,
        checkpointer: Checkpointer,
        interrupt_before: Collection[str] | Literal["*"] = (),
        interrupt_after: Collection[str] | Literal["*"] = (),
    ) -> PersistentGraph[StateT, InputT, OutputT]: ...
    def compile[ProviderT: Callable[..., object]](
        self,
        dependency_overrides: Mapping[ProviderT, Callable[..., object]] | None = None,
        *,
        checkpointer: Checkpointer | None = None,
        interrupt_before: Collection[str] | Literal["*"] = (),
        interrupt_after: Collection[str] | Literal["*"] = (),
    ) -> CompiledGraph[StateT, InputT, OutputT] | PersistentGraph[StateT, InputT, OutputT]:
        """Checks the graph and returns a graph that runs.

        A `checkpointer` selects a `PersistentGraph`, which saves a checkpoint after each step.
        A run stops before each node in `interrupt_before` and after each node in `interrupt_after`.
        `"*"` selects each node.
        """
        graph = self.compile_graph(dependency_overrides, list_supplied(persistent=checkpointer is not None))
        if checkpointer is None:
            receivers = graph.find_receivers()
            if receivers:
                raise ValueError(
                    f"the nodes {receivers!r} receive a message, and only a graph with a checkpointer can wait for one"
                )
            return graph
        names = frozenset(graph.nodes)
        before = parse_breakpoints(interrupt_before, names)
        after = parse_breakpoints(interrupt_after, names)
        unknown = sorted((before | after) - names)
        if unknown:
            raise ValueError(f"interrupt_before or interrupt_after refer to {unknown!r}, which are not nodes")
        return PersistentGraph(graph, checkpointer, Breakpoints(before, after))

    def compile_graph[ProviderT: Callable[..., object]](
        self,
        dependency_overrides: Mapping[ProviderT, Callable[..., object]] | None,
        supplied: Mapping[Supplied, Scope],
    ) -> CompiledGraph[StateT, InputT, OutputT]:
        overrides: dict[Callable[..., object], Callable[..., object]] = (
            {} if dependency_overrides is None else dict(dependency_overrides.items())
        )
        endpoints = {
            *(endpoint for edge in self.edges for endpoint in (edge.source, edge.target) if isinstance(endpoint, str)),
            *(
                endpoint
                for join in self.joins
                for endpoint in (*join.sources, join.target)
                if isinstance(endpoint, str)
            ),
        }
        declared = {target for node in self.nodes.values() for target in read_declared_targets(node)}
        unknown = sorted((endpoints | declared) - self.nodes.keys() - self.child_graphs.keys())
        if unknown:
            raise ValueError(f"edges or declared targets refer to {unknown!r}, which are not nodes of this graph")
        entry_nodes = frozenset(
            edge.target for edge in self.edges if edge.source is Endpoint.start and isinstance(edge.target, str)
        )
        if not entry_nodes:
            raise ValueError("the graph has no edge from Endpoint.start to a node")
        input_schemas: dict[str, type[object]] = {
            **{name: self.node_input_schemas.get(name, self.state_schema) for name in self.nodes},
            **{name: child.input_schema for name, child in self.child_graphs.items()},
        }
        actions: dict[str, CallPlan | CompiledGraph[object, Never, object]] = {
            **{name: plan_call(node, overrides, supplied) for name, node in self.nodes.items()},
            **{name: child.compile_graph(overrides, supplied) for name, child in self.child_graphs.items()},
        }
        waiting_deferred = sorted(
            name
            for name, action in actions.items()
            if name in self.deferred and isinstance(action, CallPlan) and action.received is not None
        )
        if waiting_deferred:
            raise ValueError(
                f"the nodes {waiting_deferred!r} receive a message and are deferred, "
                f"and a deferred node runs only when no other node waits"
            )
        join_keys = sorted(
            (JoinKey(join.sources, join.target) for join in self.joins if isinstance(join.target, str)),
            key=lambda key: sorted(key.sources),
        )
        return CompiledGraph(
            output_schema=self.output_schema,
            channels=merge_channels(
                [self.state_schema, self.input_schema, self.output_schema, *input_schemas.values()]
            ),
            output_fields=frozenset(build_channels(self.output_schema)),
            nodes={
                name: CompiledNode(
                    action=action,
                    input_schema=input_schemas[name],
                    input_fields=frozenset(build_channels(input_schemas[name])),
                    targets=frozenset(
                        edge.target for edge in self.edges if edge.source == name and isinstance(edge.target, str)
                    ),
                    joins=frozenset(key for key in join_keys if name in key.sources),
                    defer=name in self.deferred,
                    triggers=(BranchKey(name), *(key for key in join_keys if key.target == name)),
                    message=read_message_schema(action),
                )
                for name, action in actions.items()
            },
            entry_nodes=entry_nodes,
        )


def read_message_schema(action: CallPlan | CompiledGraph[object, Never, object]) -> MessageSchema | None:
    match action:
        case CompiledGraph():
            return None
        case CallPlan():
            annotation = action.received
            return None if annotation is None else MessageSchema(get_args(annotation)[0], TypeAdapter(annotation))
        case _:
            assert_never(action)


def parse_breakpoints(nodes: Collection[str] | Literal["*"], names: frozenset[str]) -> frozenset[str]:
    match nodes:
        case "*":
            return names
        case str():
            raise TypeError(
                f"interrupt_before and interrupt_after take '*' or a collection of node names, got {nodes!r}"
            )
        case Collection():
            return frozenset(nodes)
        case _:
            assert_never(nodes)


def list_supplied(*, persistent: bool) -> Mapping[Supplied, Scope]:
    supplied: dict[Supplied, Scope] = {
        NodeValue.state: Scope.node,
        NodeValue.received: Scope.node,
        StreamWriter: Scope.node,
    }
    if persistent:
        supplied |= {RunTarget: Scope.run, ShutdownSignal: Scope.run}
    return supplied

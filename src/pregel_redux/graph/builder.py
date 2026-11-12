from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from typing import Any, Concatenate, Literal, Never, Protocol, overload, runtime_checkable

from pregel_redux.channels import build_channels, merge_channels
from pregel_redux.checkpoints import Checkpointer
from pregel_redux.commands import Endpoint, read_declared_targets
from pregel_redux.dependencies import CallPlan, plan_call
from pregel_redux.graph.compiled import CompiledGraph, CompiledNode
from pregel_redux.graph.persistent import PersistentGraph
from pregel_redux.pregel import BranchKey, JoinKey

type Node[InputT] = Callable[Concatenate[InputT, ...], object]


@runtime_checkable
class GraphBuilder(Protocol):
    @property
    def input_schema(self) -> type[object]: ...
    @property
    def dependencies(self) -> frozenset[Callable[..., object]]: ...
    def compile(
        self, dependency_overrides: Mapping[Callable[..., object], Callable[..., object]] | None = None
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
        if name in self.nodes or name in self.child_graphs:
            raise ValueError(f"node {name!r} is already in this graph")
        if name == "__start__":
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
        match source:
            case tuple():
                self.joins.add(Join(frozenset(source), target))
            case _:
                self.edges.add(Edge(source, target))

    @property
    def dependencies(self) -> frozenset[Callable[..., object]]:
        plans = [plan_call(node, {}) for node in self.nodes.values()]
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
        graph = self.compile_graph(dependency_overrides)
        if checkpointer is None:
            return graph
        names = frozenset(graph.nodes)
        before = names if interrupt_before == "*" else frozenset(interrupt_before)
        after = names if interrupt_after == "*" else frozenset(interrupt_after)
        unknown = sorted((before | after) - names)
        if unknown:
            raise ValueError(f"interrupt_before or interrupt_after refer to {unknown!r}, which are not nodes")
        return PersistentGraph(replace(graph, interrupt_before=before, interrupt_after=after), checkpointer)

    def compile_graph[ProviderT: Callable[..., object]](
        self, dependency_overrides: Mapping[ProviderT, Callable[..., object]] | None
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
            **{name: plan_call(node, overrides) for name, node in self.nodes.items()},
            **{name: child.compile(overrides) for name, child in self.child_graphs.items()},
        }
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
                )
                for name, action in actions.items()
            },
            entry_nodes=entry_nodes,
            interrupt_before=frozenset(),
            interrupt_after=frozenset(),
        )

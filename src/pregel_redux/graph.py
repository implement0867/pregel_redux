import asyncio
from collections import ChainMap
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Concatenate, Literal, Never, Protocol, assert_never, overload, runtime_checkable

from pregel_redux.channels import Channel, apply_writes, build_channels, extract_writes, merge_channels
from pregel_redux.commands import Endpoint, Outcome, Send, read_declared_targets, read_outcome
from pregel_redux.dependencies import CallPlan, Depends, execute_plan, open_providers, plan_call

type Node[InputT] = Callable[Concatenate[InputT, ...], object]


@runtime_checkable
class GraphBuilder(Protocol):
    @property
    def input_schema(self) -> Callable[..., object]: ...
    @property
    def dependencies(self) -> frozenset[Callable[..., object]]: ...
    def compile(
        self, dependency_overrides: Mapping[Callable[..., object], Callable[..., object]] | None = None
    ) -> "CompiledGraph[object, Never, object]": ...


@dataclass(frozen=True)
class Run:
    values: dict[Depends, object]
    exits: AsyncExitStack
    lock: asyncio.Lock
    step_limit: int


@dataclass(frozen=True)
class CompiledNode:
    action: "CallPlan | CompiledGraph[object, Never, object]"
    input_schema: Callable[..., object]
    input_fields: frozenset[str]
    targets: frozenset[str]

    def build_input(self, values: Mapping[str, object]) -> object:
        return self.input_schema(**{name: values[name] for name in self.input_fields if name in values})

    def build_send_input(self, send: Send) -> object:
        payload = send.arg
        match payload:
            case Mapping():
                fields: dict[str, object] = {**payload}
                return self.input_schema(**fields)
            case _:
                return payload


@dataclass(frozen=True)
class NodeCall:
    node: CompiledNode
    node_input: object


@dataclass(frozen=True)
class Edge:
    source: str | Literal[Endpoint.start]
    target: str | Literal[Endpoint.end]


@dataclass(frozen=True)
class CompiledGraph[StateT, InputT, OutputT]:
    output_schema: Callable[..., OutputT]
    channels: Mapping[str, Channel]
    output_fields: frozenset[str]
    nodes: Mapping[str, CompiledNode]
    entry_nodes: frozenset[str]

    async def ainvoke(self, input: InputT, *, step_limit: int = 25) -> OutputT:
        async with AsyncExitStack() as run_exits:
            return await self.run_steps(input, Run({}, run_exits, asyncio.Lock(), step_limit))

    def invoke(self, input: InputT, *, step_limit: int = 25) -> OutputT:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input, step_limit=step_limit))
        raise RuntimeError("invoke cannot run inside a running event loop, so await ainvoke there")

    async def run_steps(self, input: object, run: Run) -> OutputT:
        values = apply_writes(self.channels, {}, [extract_writes(input)])
        names = self.entry_nodes
        sends: tuple[Send, ...] = ()
        steps = 0
        while names or sends:
            if steps == run.step_limit:
                raise RuntimeError(f"the graph did not finish within {run.step_limit} steps")
            steps += 1
            calls = [
                *(NodeCall(self.nodes[name], self.nodes[name].build_input(values)) for name in sorted(names)),
                *(NodeCall(self.nodes[send.node], self.nodes[send.node].build_send_input(send)) for send in sends),
            ]
            async with AsyncExitStack() as step_exits:
                outcomes = await self.run_step(calls, run, step_exits)
                values = apply_writes(self.channels, values, [outcome.writes for outcome in outcomes])
            names = frozenset[str]().union(*(outcome.routes.targets for outcome in outcomes))
            sends = tuple(send for outcome in outcomes for send in outcome.routes.sends)
        return self.output_schema(**{name: values[name] for name in self.output_fields if name in values})

    async def run_step(self, calls: Sequence[NodeCall], run: Run, step_exits: AsyncExitStack) -> list[Outcome]:
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self.run_node(call, run, step_exits)) for call in calls]
        except ExceptionGroup as failures:
            if len(failures.exceptions) == 1:
                raise failures.exceptions[0] from None
            raise
        return [task.result() for task in tasks]

    async def run_node(self, call: NodeCall, run: Run, step_exits: AsyncExitStack) -> Outcome:
        match call.node.action:
            case CallPlan() as plan:
                return await self.call_function(plan, call, run, step_exits)
            case CompiledGraph() as child:
                result = await child.run_steps(call.node_input, run)
                return read_outcome(result, call.node.targets, self.nodes.keys())
            case _:
                assert_never(call.node.action)

    async def call_function(self, plan: CallPlan, call: NodeCall, run: Run, step_exits: AsyncExitStack) -> Outcome:
        if plan.run_providers:
            async with run.lock:
                await open_providers(plan.run_providers, run.values, run.exits)
        node_values = ChainMap[Depends, object]({}, dict(run.values))
        outcome: Outcome | None = None
        async with AsyncExitStack[bool]() as node_exits:
            result = await execute_plan(plan, call.node_input, node_values, node_exits)
            outcome = read_outcome(result, call.node.targets, self.nodes.keys())
            step_exits.push_async_callback(node_exits.pop_all().aclose)
        if outcome is None:
            raise RuntimeError(f"a dependency of {plan.function!r} suppressed the exception of its node")
        return outcome


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
        self.node_input_schemas: dict[str, Callable[..., object]] = {}
        self.child_graphs: dict[str, GraphBuilder] = {}
        self.edges: set[Edge] = set()

    @overload
    def add_node(self, name: str, node: Node[StateT]) -> None: ...
    @overload
    def add_node[NodeInputT](self, name: str, node: Node[NodeInputT], *, input_schema: type[NodeInputT]) -> None: ...
    @overload
    def add_node[ChildStateT, ChildInputT, ChildOutputT](
        self, name: str, node: "StateGraph[ChildStateT, ChildInputT, ChildOutputT]"
    ) -> None: ...
    def add_node(
        self,
        name: str,
        node: Callable[..., object] | GraphBuilder,
        *,
        input_schema: Callable[..., object] | None = None,
    ) -> None:
        if name in self.nodes or name in self.child_graphs:
            raise ValueError(f"node {name!r} is already in this graph")
        match node:
            case GraphBuilder():
                self.child_graphs[name] = node
            case _:
                self.nodes[name] = node
                if input_schema is not None:
                    self.node_input_schemas[name] = input_schema

    def add_edge(self, source: str | Literal[Endpoint.start], target: str | Literal[Endpoint.end]) -> None:
        self.edges.add(Edge(source, target))

    @property
    def dependencies(self) -> frozenset[Callable[..., object]]:
        plans = [plan_call(node, {}) for node in self.nodes.values()]
        return frozenset(
            item.request.dependency for plan in plans for item in (*plan.run_providers, *plan.node_providers)
        ).union(*(child.dependencies for child in self.child_graphs.values()))

    def compile[ProviderT: Callable[..., object]](
        self, dependency_overrides: Mapping[ProviderT, Callable[..., object]] | None = None
    ) -> CompiledGraph[StateT, InputT, OutputT]:
        overrides: dict[Callable[..., object], Callable[..., object]] = (
            {} if dependency_overrides is None else dict(dependency_overrides.items())
        )
        endpoints = {
            endpoint for edge in self.edges for endpoint in (edge.source, edge.target) if isinstance(endpoint, str)
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
        input_schemas: dict[str, Callable[..., object]] = {
            **{name: self.node_input_schemas.get(name, self.state_schema) for name in self.nodes},
            **{name: child.input_schema for name, child in self.child_graphs.items()},
        }
        actions: dict[str, CallPlan | CompiledGraph[object, Never, object]] = {
            **{name: plan_call(node, overrides) for name, node in self.nodes.items()},
            **{name: child.compile(overrides) for name, child in self.child_graphs.items()},
        }
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
                )
                for name, action in actions.items()
            },
            entry_nodes=entry_nodes,
        )

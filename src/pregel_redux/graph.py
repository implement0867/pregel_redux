import asyncio
from collections import ChainMap
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import Enum
from typing import Any, Concatenate, Literal, overload

from pregel_redux.channels import Channel, apply_writes, build_channels, extract_writes, merge_channels
from pregel_redux.dependencies import CallPlan, Depends, execute_plan, open_providers, plan_call


class Endpoint(Enum):
    start = "start"
    end = "end"


type Node[StateT] = Callable[Concatenate[StateT, ...], object]
type Route = str | Literal[Endpoint.end] | Sequence[str | Literal[Endpoint.end]]
type Router[StateT] = Callable[Concatenate[StateT, ...], Route | Awaitable[Route]]


@dataclass(frozen=True)
class CompiledNode:
    action: CallPlan
    routers: tuple[CallPlan, ...]
    targets: frozenset[str]


@dataclass(frozen=True)
class NodeResult:
    writes: Mapping[str, object]
    targets: frozenset[str]


@dataclass(frozen=True)
class CompiledGraph[StateT, InputT, OutputT]:
    state_schema: Callable[..., StateT]
    output_schema: Callable[..., OutputT]
    channels: Mapping[str, Channel]
    state_fields: frozenset[str]
    output_fields: frozenset[str]
    nodes: Mapping[str, CompiledNode]
    entry_nodes: frozenset[str]

    async def ainvoke(self, input: InputT, *, step_limit: int = 25) -> OutputT:
        values = apply_writes(self.channels, {}, [extract_writes(input)])
        scheduled = self.entry_nodes
        run_values: dict[Depends, object] = {}
        steps = 0
        async with AsyncExitStack() as run_exits:
            while scheduled:
                if steps == step_limit:
                    raise RuntimeError(f"the graph did not finish within {step_limit} steps")
                steps += 1
                names = sorted(scheduled)
                for name in names:
                    for plan in (self.nodes[name].action, *self.nodes[name].routers):
                        await open_providers(plan.run_providers, run_values, run_exits)
                results = await self.run_step(names, values, run_values)
                values = apply_writes(self.channels, values, [result.writes for result in results])
                scheduled = frozenset[str]().union(*(result.targets for result in results))
        return self.output_schema(**{name: values[name] for name in self.output_fields if name in values})

    async def run_step(
        self, names: Sequence[str], values: Mapping[str, object], run_values: Mapping[Depends, object]
    ) -> list[NodeResult]:
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self.run_node(self.nodes[name], values, run_values)) for name in names]
        except ExceptionGroup as failures:
            if len(failures.exceptions) == 1:
                raise failures.exceptions[0] from None
            raise
        return [task.result() for task in tasks]

    async def run_node(
        self, node: CompiledNode, values: Mapping[str, object], run_values: Mapping[Depends, object]
    ) -> NodeResult:
        node_values = ChainMap[Depends, object]({}, dict(run_values))
        async with AsyncExitStack() as node_exits:
            update = await execute_plan(node.action, self.build_state(values), node_values, node_exits)
            writes = extract_writes(update)
            written = apply_writes(self.channels, values, [writes])
            routes = [
                await execute_plan(router, self.build_state(written), node_values, node_exits)
                for router in node.routers
            ]
        return NodeResult(writes, node.targets.union(*(self.resolve_targets(route) for route in routes)))

    def build_state(self, values: Mapping[str, object]) -> StateT:
        return self.state_schema(**{name: values[name] for name in self.state_fields if name in values})

    def resolve_targets(self, route: object) -> frozenset[str]:
        selections: Sequence[object]
        match route:
            case str() | Endpoint.end:
                selections = [route]
            case Sequence():
                selections = route
            case _:
                raise TypeError(f"a router returned {route!r}, which is not a node name, Endpoint.end or a sequence")
        targets = frozenset(name for name in self.nodes if name in selections)
        unknown = [selection for selection in selections if selection is not Endpoint.end and selection not in targets]
        if unknown:
            raise ValueError(f"a router returned {unknown!r}, which are not nodes of this graph")
        return targets


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
        self.nodes: dict[str, Node[StateT]] = {}
        self.routers: dict[str, list[Router[StateT]]] = {}
        self.edges: set[tuple[str | Endpoint, str | Endpoint]] = set()

    def add_node(self, name: str, node: Node[StateT]) -> None:
        if name in self.nodes:
            raise ValueError(f"node {name!r} is already in this graph")
        self.nodes[name] = node

    def add_edge(self, source: str | Literal[Endpoint.start], target: str | Literal[Endpoint.end]) -> None:
        self.edges.add((source, target))

    def add_conditional_edges(self, source: str, router: Router[StateT]) -> None:
        self.routers.setdefault(source, []).append(router)

    @property
    def dependencies(self) -> frozenset[Callable[..., object]]:
        functions = [*self.nodes.values(), *(router for routers in self.routers.values() for router in routers)]
        plans = [plan_call(function, {}) for function in functions]
        return frozenset(
            item.request.dependency for plan in plans for item in (*plan.run_providers, *plan.node_providers)
        )

    def compile[ProviderT: Callable[..., object]](
        self, dependency_overrides: Mapping[ProviderT, Callable[..., object]] | None = None
    ) -> CompiledGraph[StateT, InputT, OutputT]:
        overrides: dict[Callable[..., object], Callable[..., object]] = (
            {} if dependency_overrides is None else dict(dependency_overrides.items())
        )
        endpoints = {endpoint for edge in self.edges for endpoint in edge if isinstance(endpoint, str)}
        unknown = sorted((endpoints | self.routers.keys()) - self.nodes.keys())
        if unknown:
            raise ValueError(f"edges refer to {unknown!r}, which are not nodes of this graph")
        entry_nodes = frozenset(
            target for source, target in self.edges if source is Endpoint.start and isinstance(target, str)
        )
        if not entry_nodes:
            raise ValueError("the graph has no edge from Endpoint.start to a node")
        return CompiledGraph(
            state_schema=self.state_schema,
            output_schema=self.output_schema,
            channels=merge_channels([self.state_schema, self.input_schema, self.output_schema]),
            state_fields=frozenset(build_channels(self.state_schema)),
            output_fields=frozenset(build_channels(self.output_schema)),
            nodes={
                name: CompiledNode(
                    action=plan_call(node, overrides),
                    routers=tuple(plan_call(router, overrides) for router in self.routers.get(name, [])),
                    targets=frozenset(
                        target for source, target in self.edges if source == name and isinstance(target, str)
                    ),
                )
                for name, node in self.nodes.items()
            },
            entry_nodes=entry_nodes,
        )

import asyncio
from collections import ChainMap
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Concatenate, Literal, overload

from pregel_redux.channels import Channel, apply_writes, build_channels, extract_writes, merge_channels
from pregel_redux.commands import Endpoint, Outcome, Send, read_declared_targets, read_outcome
from pregel_redux.dependencies import CallPlan, Depends, execute_plan, open_providers, plan_call

type Node[InputT] = Callable[Concatenate[InputT, ...], object]


@dataclass(frozen=True)
class CompiledNode:
    action: CallPlan
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
class CompiledGraph[StateT, InputT, OutputT]:
    output_schema: Callable[..., OutputT]
    channels: Mapping[str, Channel]
    output_fields: frozenset[str]
    nodes: Mapping[str, CompiledNode]
    entry_nodes: frozenset[str]

    async def ainvoke(self, input: InputT, *, step_limit: int = 25) -> OutputT:
        values = apply_writes(self.channels, {}, [extract_writes(input)])
        names = self.entry_nodes
        sends: tuple[Send, ...] = ()
        run_values: dict[Depends, object] = {}
        steps = 0
        async with AsyncExitStack() as run_exits:
            while names or sends:
                if steps == step_limit:
                    raise RuntimeError(f"the graph did not finish within {step_limit} steps")
                steps += 1
                calls = [
                    *((self.nodes[name], self.nodes[name].build_input(values)) for name in sorted(names)),
                    *((self.nodes[send.node], self.nodes[send.node].build_send_input(send)) for send in sends),
                ]
                for node, _ in calls:
                    await open_providers(node.action.run_providers, run_values, run_exits)
                async with AsyncExitStack() as step_exits:
                    outcomes = await self.run_step(calls, run_values, step_exits)
                    values = apply_writes(self.channels, values, [outcome.writes for outcome in outcomes])
                names = frozenset[str]().union(*(outcome.targets for outcome in outcomes))
                sends = tuple(send for outcome in outcomes for send in outcome.sends)
        return self.output_schema(**{name: values[name] for name in self.output_fields if name in values})

    async def run_step(
        self,
        calls: Sequence[tuple[CompiledNode, object]],
        run_values: Mapping[Depends, object],
        step_exits: AsyncExitStack,
    ) -> list[Outcome]:
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self.run_node(node, node_input, run_values, step_exits))
                    for node, node_input in calls
                ]
        except ExceptionGroup as failures:
            if len(failures.exceptions) == 1:
                raise failures.exceptions[0] from None
            raise
        return [task.result() for task in tasks]

    async def run_node(
        self,
        node: CompiledNode,
        node_input: object,
        run_values: Mapping[Depends, object],
        step_exits: AsyncExitStack,
    ) -> Outcome:
        node_values = ChainMap[Depends, object]({}, dict(run_values))
        outcome: Outcome | None = None
        async with AsyncExitStack[bool]() as node_exits:
            result = await execute_plan(node.action, node_input, node_values, node_exits)
            outcome = read_outcome(result, node.targets, self.nodes.keys())
            step_exits.push_async_callback(node_exits.pop_all().aclose)
        if outcome is None:
            raise RuntimeError(f"a dependency of {node.action.function!r} suppressed the exception of its node")
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
        self.edges: set[tuple[str | Endpoint, str | Endpoint]] = set()

    @overload
    def add_node(self, name: str, node: Node[StateT]) -> None: ...
    @overload
    def add_node[NodeInputT](self, name: str, node: Node[NodeInputT], *, input_schema: type[NodeInputT]) -> None: ...
    def add_node(
        self, name: str, node: Callable[..., object], *, input_schema: Callable[..., object] | None = None
    ) -> None:
        if name in self.nodes:
            raise ValueError(f"node {name!r} is already in this graph")
        self.nodes[name] = node
        if input_schema is not None:
            self.node_input_schemas[name] = input_schema

    def add_edge(self, source: str | Literal[Endpoint.start], target: str | Literal[Endpoint.end]) -> None:
        self.edges.add((source, target))

    @property
    def dependencies(self) -> frozenset[Callable[..., object]]:
        plans = [plan_call(node, {}) for node in self.nodes.values()]
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
        declared = {target for node in self.nodes.values() for target in read_declared_targets(node)}
        unknown = sorted((endpoints | declared) - self.nodes.keys())
        if unknown:
            raise ValueError(f"edges or declared targets refer to {unknown!r}, which are not nodes of this graph")
        input_schemas = {name: self.node_input_schemas.get(name, self.state_schema) for name in self.nodes}
        entry_nodes = frozenset(
            target for source, target in self.edges if source is Endpoint.start and isinstance(target, str)
        )
        if not entry_nodes:
            raise ValueError("the graph has no edge from Endpoint.start to a node")
        return CompiledGraph(
            output_schema=self.output_schema,
            channels=merge_channels(
                [self.state_schema, self.input_schema, self.output_schema, *self.node_input_schemas.values()]
            ),
            output_fields=frozenset(build_channels(self.output_schema)),
            nodes={
                name: CompiledNode(
                    action=plan_call(node, overrides),
                    input_schema=input_schemas[name],
                    input_fields=frozenset(build_channels(input_schemas[name])),
                    targets=frozenset(
                        target for source, target in self.edges if source == name and isinstance(target, str)
                    ),
                )
                for name, node in self.nodes.items()
            },
            entry_nodes=entry_nodes,
        )

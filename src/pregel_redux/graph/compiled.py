import asyncio
import inspect
from collections import ChainMap
from collections.abc import Callable, Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from contextvars import copy_context
from dataclasses import dataclass
from functools import cached_property
from typing import Never, Protocol, assert_never

from pydantic import TypeAdapter

from pregel_redux.channels import Channel, extract_writes, reject_unknown_channels
from pregel_redux.commands import Send, read_outcome
from pregel_redux.dependencies import CallPlan, Depends, execute_plan, open_providers
from pregel_redux.pregel import (
    Arrival,
    JoinKey,
    Position,
    PullTask,
    PushTask,
    Task,
    TaskWrites,
    TriggerKey,
    advance,
    plan_tasks,
    start_position,
)
from pregel_redux.settings import RunSettings


@dataclass(frozen=True)
class Run:
    values: dict[Depends, object]
    exits: AsyncExitStack
    lock: asyncio.Lock
    step_limit: int


@dataclass(frozen=True)
class CompiledNode:
    action: "CallPlan | CompiledGraph[object, Never, object]"
    input_schema: type[object]
    input_fields: frozenset[str]
    targets: frozenset[str]
    joins: frozenset[JoinKey]
    triggers: tuple[TriggerKey, ...]

    @cached_property
    def input_adapter(self) -> TypeAdapter[object]:
        return TypeAdapter(self.input_schema)

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
    name: str
    send_index: int | None
    node_input: object


class Journal(Protocol):
    async def save(self, position: Position) -> None: ...
    async def enter_child(
        self, child: "CompiledGraph[object, Never, object]", call: NodeCall
    ) -> "tuple[Position, Journal]": ...


class NoJournal:
    async def save(self, position: Position) -> None:
        pass

    async def enter_child(
        self, child: "CompiledGraph[object, Never, object]", call: NodeCall
    ) -> "tuple[Position, Journal]":
        return child.start(extract_writes(call.node_input), {}), self


@dataclass(frozen=True)
class CompiledGraph[StateT, InputT, OutputT]:
    output_schema: Callable[..., OutputT]
    channels: Mapping[str, Channel]
    output_fields: frozenset[str]
    nodes: Mapping[str, CompiledNode]
    entry_nodes: frozenset[str]

    async def ainvoke(self, input: InputT, *, step_limit: int | None = None) -> OutputT:
        return await self.run(self.start({}, extract_writes(input)), step_limit, NoJournal())

    async def run(self, position: Position, step_limit: int | None, journal: Journal) -> OutputT:
        limit = RunSettings().step_limit if step_limit is None else step_limit
        async with AsyncExitStack() as run_exits:
            return await self.run_steps(position, Run({}, run_exits, asyncio.Lock(), limit), journal)

    def find_async_nodes(self) -> list[str]:
        names: list[str] = []
        for name, node in self.nodes.items():
            match node.action:
                case CallPlan() as plan:
                    if inspect.iscoroutinefunction(plan.function):
                        names.append(name)
                case CompiledGraph() as child:
                    names.extend(f"{name}.{child_name}" for child_name in child.find_async_nodes())
                case _:
                    assert_never(node.action)
        return sorted(names)

    def run_in_new_loop[ResultT](
        self, start_run: Callable[[], Coroutine[object, object, ResultT]], async_method: str
    ) -> ResultT:
        async_nodes = self.find_async_nodes()
        if async_nodes:
            raise TypeError(f"nodes {async_nodes!r} have only an async function, so call {async_method}")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(start_run())
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(copy_context().run, asyncio.run, start_run()).result()

    def invoke(self, input: InputT, *, step_limit: int | None = None) -> OutputT:
        return self.run_in_new_loop(lambda: self.ainvoke(input, step_limit=step_limit), "ainvoke")

    def start(self, given: Mapping[str, object], update: Mapping[str, object]) -> Position:
        reject_unknown_channels(self.channels, given)
        initial = {
            name: channel.reducer.make_initial()
            for name, channel in self.channels.items()
            if channel.reducer is not None
        }
        values = {name: self.channels[name].parse(value) for name, value in {**initial, **given}.items()}
        joins = {key for node in self.nodes.values() for key in node.joins}
        entry = TaskWrites(fields=update, branches=self.entry_nodes, arrivals=(), sends=())
        return advance(self.channels, start_position(values, self.nodes.keys(), joins), [], [entry])

    def build_call(self, task: Task, values: Mapping[str, object]) -> NodeCall:
        match task:
            case PullTask():
                return NodeCall(task.node, None, self.nodes[task.node].build_input(values))
            case PushTask():
                return NodeCall(task.send.node, task.index, self.nodes[task.send.node].build_send_input(task.send))
            case _:
                assert_never(task)

    async def run_steps(self, start: Position, run: Run, journal: Journal) -> OutputT:
        position = start
        triggers = {name: node.triggers for name, node in self.nodes.items()}
        steps = 0
        while tasks := plan_tasks(position, triggers):
            if steps == run.step_limit:
                raise RuntimeError(f"the graph did not finish within {run.step_limit} steps")
            steps += 1
            calls = [self.build_call(task, position.channels.values) for task in tasks]
            async with AsyncExitStack() as step_exits:
                writes = await self.run_step(calls, run, journal, step_exits)
                position = advance(self.channels, position, tasks, writes)
                await journal.save(position)
        values = position.channels.values
        return self.output_schema(**{name: values[name] for name in self.output_fields if name in values})

    async def run_step(
        self, calls: Sequence[NodeCall], run: Run, journal: Journal, step_exits: AsyncExitStack
    ) -> list[TaskWrites]:
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self.run_node(call, run, journal, step_exits)) for call in calls]
        except ExceptionGroup as failures:
            if len(failures.exceptions) == 1:
                raise failures.exceptions[0] from None
            raise
        return [task.result() for task in tasks]

    async def run_node(self, call: NodeCall, run: Run, journal: Journal, step_exits: AsyncExitStack) -> TaskWrites:
        node = self.nodes[call.name]
        match node.action:
            case CallPlan() as plan:
                return await self.call_function(plan, call, run, step_exits)
            case CompiledGraph() as child:
                start, child_journal = await journal.enter_child(child, call)
                result = await child.run_steps(start, run, child_journal)
                return self.read_writes(call.name, result)
            case _:
                assert_never(node.action)

    def read_writes(self, name: str, result: object) -> TaskWrites:
        node = self.nodes[name]
        outcome = read_outcome(result, node.targets, self.nodes.keys())
        return TaskWrites(
            fields=outcome.writes,
            branches=outcome.routes.targets,
            arrivals=tuple(Arrival(key, name) for key in node.joins),
            sends=outcome.routes.sends,
        )

    async def call_function(self, plan: CallPlan, call: NodeCall, run: Run, step_exits: AsyncExitStack) -> TaskWrites:
        if plan.run_providers:
            async with run.lock:
                await open_providers(plan.run_providers, run.values, run.exits)
        node_values = ChainMap[Depends, object]({}, dict(run.values))
        writes: TaskWrites | None = None
        async with AsyncExitStack[bool]() as node_exits:
            result = await execute_plan(plan, call.node_input, node_values, node_exits)
            writes = self.read_writes(call.name, result)
            step_exits.push_async_callback(node_exits.pop_all().aclose)
        if writes is None:
            raise RuntimeError(f"a dependency of {plan.function!r} suppressed the exception of its node")
        return writes

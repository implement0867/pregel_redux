import asyncio
import inspect
from collections import ChainMap
from collections.abc import Callable, Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from contextvars import copy_context
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Concatenate, Literal, Never, Protocol, assert_never, overload, runtime_checkable
from uuid import UUID, uuid4

from pydantic import TypeAdapter

from pregel_redux.channels import (
    Channel,
    apply_writes,
    build_channels,
    extract_writes,
    merge_channels,
    reject_unknown_channels,
)
from pregel_redux.checkpoints import CheckpointSource, InterruptedTask, TaskAddress
from pregel_redux.commands import Endpoint, read_declared_targets, read_outcome
from pregel_redux.dependencies import (
    CallPlan,
    Depends,
    execute_plan,
    finish_despite_cancellation,
    open_providers,
    plan_call,
)
from pregel_redux.interrupts import (
    Interrupt,
    InterruptId,
    NodeInterrupt,
    TaskInterrupts,
    current_interrupts,
    find_interruption,
)

type Node[InputT] = Callable[Concatenate[InputT, ...], object]


@runtime_checkable
class GraphBuilder(Protocol):
    @property
    def input_schema(self) -> type[object]: ...
    @property
    def dependencies(self) -> frozenset[Callable[..., object]]: ...
    def compile(
        self, dependency_overrides: Mapping[Callable[..., object], Callable[..., object]] | None = None
    ) -> "CompiledGraph[object, Never, object]": ...


@dataclass(frozen=True)
class PullTask:
    node: str


@dataclass(frozen=True)
class PushTask:
    node: str
    payload: object


type Task = PullTask | PushTask


@dataclass(frozen=True)
class TaskOutput:
    writes: Mapping[str, object]
    targets: frozenset[str]
    pushes: tuple[PushTask, ...]


@dataclass(frozen=True)
class Finished[OutputT]:
    output: OutputT


@dataclass(frozen=True)
class Interrupted:
    interrupts: tuple[Interrupt, ...]


type RunResult[OutputT] = Finished[OutputT] | Interrupted


@dataclass(frozen=True)
class Position:
    checkpoint_id: UUID
    parent_id: UUID | None
    caller: TaskAddress | None
    step: int
    source: CheckpointSource
    values: Mapping[str, object]
    tasks: tuple[Task, ...]
    progress: Mapping[int, TaskOutput | InterruptedTask]

    def follow(self, source: CheckpointSource, values: Mapping[str, object], tasks: tuple[Task, ...]) -> "Position":
        return Position(uuid4(), self.checkpoint_id, self.caller, self.step + 1, source, values, tasks, {})


def plan_tasks(outputs: Sequence[TaskOutput]) -> tuple[Task, ...]:
    targets = sorted(frozenset[str]().union(*(output.targets for output in outputs)))
    return (*(PullTask(name) for name in targets), *(push for output in outputs for push in output.pushes))


def interrupt_task(task: TaskAddress, node: str, state: InterruptedTask) -> Interrupted:
    interrupt_id = InterruptId(task=task, index=len(state.answers))
    return Interrupted((Interrupt(interrupt_id=interrupt_id, node=node, payload=state.payload),))


class Journal(Protocol):
    async def save(self, graph: "CompiledGraph[object, Never, object]", position: Position) -> None: ...
    async def record_output(
        self, graph: "CompiledGraph[object, Never, object]", task: TaskAddress, output: TaskOutput
    ) -> TaskOutput: ...
    async def record_interrupt(self, task: TaskAddress, state: InterruptedTask) -> None: ...
    async def load_latest(
        self, graph: "CompiledGraph[object, Never, object]", caller: TaskAddress | None
    ) -> Position | None: ...


class Unrecorded:
    async def save(self, graph: "CompiledGraph[object, Never, object]", position: Position) -> None:
        pass

    async def record_output(
        self, graph: "CompiledGraph[object, Never, object]", task: TaskAddress, output: TaskOutput
    ) -> TaskOutput:
        return output

    async def record_interrupt(self, task: TaskAddress, state: InterruptedTask) -> None:
        pass

    async def load_latest(
        self, graph: "CompiledGraph[object, Never, object]", caller: TaskAddress | None
    ) -> Position | None:
        return None


@dataclass(frozen=True)
class Run:
    values: dict[Depends, object]
    exits: AsyncExitStack
    lock: asyncio.Lock
    step_limit: int
    journal: Journal
    answers: Mapping[TaskAddress, tuple[str, ...]]


@dataclass(frozen=True)
class CompiledNode:
    action: "CallPlan | CompiledGraph[object, Never, object]"
    input_schema: type[object]
    input_fields: frozenset[str]
    targets: frozenset[str]

    @cached_property
    def input_adapter(self) -> TypeAdapter[object]:
        return TypeAdapter(self.input_schema)

    def build_input(self, values: Mapping[str, object]) -> object:
        return self.input_schema(**{name: values[name] for name in self.input_fields if name in values})

    def build_send_input(self, payload: object) -> object:
        match payload:
            case Mapping():
                fields: dict[str, object] = {**payload}
                return self.input_schema(**fields)
            case _:
                return payload

    def build_task_input(self, task: Task, values: Mapping[str, object]) -> object:
        match task:
            case PullTask():
                return self.build_input(values)
            case PushTask():
                return self.build_send_input(task.payload)
            case _:
                assert_never(task)


@dataclass(frozen=True)
class Edge:
    source: str | Literal[Endpoint.start]
    target: str | Literal[Endpoint.end]


def run_coroutine[ResultT](start: Callable[[], Coroutine[object, object, ResultT]]) -> ResultT:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(start())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(copy_context().run, asyncio.run, start()).result()


@dataclass(frozen=True)
class CompiledGraph[StateT, InputT, OutputT]:
    state_schema: Callable[..., StateT]
    state_fields: frozenset[str]
    output_schema: Callable[..., OutputT]
    channels: Mapping[str, Channel]
    output_fields: frozenset[str]
    nodes: Mapping[str, CompiledNode]
    entry_nodes: frozenset[str]

    async def ainvoke(self, input: InputT, *, step_limit: int = 25) -> OutputT:
        position = self.start({}, extract_writes(input), caller=None, parent=None)
        result = await self.run(position, step_limit=step_limit, journal=Unrecorded(), answers={})
        match result:
            case Finished(output=output):
                return output
            case Interrupted(interrupts=interrupts):
                raise RuntimeError(f"the run stopped at {interrupts!r}, and only a run of a Thread can resume")
            case _:
                assert_never(result)

    def invoke(self, input: InputT, *, step_limit: int = 25) -> OutputT:
        self.reject_async_nodes("ainvoke")
        return run_coroutine(lambda: self.ainvoke(input, step_limit=step_limit))

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

    def reject_async_nodes(self, async_method: str) -> None:
        async_nodes = self.find_async_nodes()
        if async_nodes:
            raise TypeError(f"nodes {async_nodes!r} have only an async function, so call {async_method}")

    def build_state(self, values: Mapping[str, object]) -> StateT:
        return self.state_schema(**{name: values[name] for name in self.state_fields if name in values})

    def build_output(self, values: Mapping[str, object]) -> OutputT:
        return self.output_schema(**{name: values[name] for name in self.output_fields if name in values})

    def start(
        self,
        given: Mapping[str, object],
        update: Mapping[str, object],
        *,
        caller: TaskAddress | None,
        parent: Position | None,
    ) -> Position:
        reject_unknown_channels(self.channels, given)
        initial = {
            name: channel.reducer.make_initial()
            for name, channel in self.channels.items()
            if channel.reducer is not None
        }
        start = {name: self.channels[name].parse(value) for name, value in {**initial, **given}.items()}
        values = apply_writes(self.channels, start, [update])
        tasks = tuple(PullTask(name) for name in sorted(self.entry_nodes))
        if parent is None:
            return Position(uuid4(), None, caller, 0, CheckpointSource.input, values, tasks, {})
        return parent.follow(CheckpointSource.input, values, tasks)

    async def enter(self, journal: Journal, caller: TaskAddress, node_input: object) -> Position:
        resumed = await journal.load_latest(self, caller)
        if resumed is not None:
            return resumed
        position = self.start(extract_writes(node_input), {}, caller=caller, parent=None)
        await journal.save(self, position)
        return position

    async def run(
        self,
        position: Position,
        *,
        step_limit: int,
        journal: Journal,
        answers: Mapping[TaskAddress, tuple[str, ...]],
    ) -> RunResult[OutputT]:
        async with AsyncExitStack() as run_exits:
            return await self.run_steps(position, Run({}, run_exits, asyncio.Lock(), step_limit, journal, answers))

    async def run_steps(self, position: Position, run: Run) -> RunResult[OutputT]:
        steps = 0
        while position.tasks:
            if steps == run.step_limit:
                raise RuntimeError(f"the graph did not finish within {run.step_limit} steps")
            steps += 1
            node_inputs = [self.nodes[task.node].build_task_input(task, position.values) for task in position.tasks]
            outputs: list[TaskOutput] = []
            interrupts: list[Interrupt] = []
            async with AsyncExitStack() as step_exits:
                for result in await self.run_step(position, node_inputs, run, step_exits):
                    match result:
                        case TaskOutput():
                            outputs.append(result)
                        case Interrupted():
                            interrupts.extend(result.interrupts)
                        case _:
                            assert_never(result)
                if interrupts:
                    return Interrupted(tuple(interrupts))
                values = apply_writes(self.channels, position.values, [output.writes for output in outputs])
            position = position.follow(CheckpointSource.loop, values, plan_tasks(outputs))
            await run.journal.save(self, position)
        return Finished(self.build_output(position.values))

    async def run_step(
        self, position: Position, node_inputs: Sequence[object], run: Run, step_exits: AsyncExitStack
    ) -> list[TaskOutput | Interrupted]:
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self.run_task(position, index, node_input, run, step_exits))
                    for index, node_input in enumerate(node_inputs)
                ]
        except ExceptionGroup as failures:
            if len(failures.exceptions) == 1:
                raise failures.exceptions[0] from None
            raise
        return [task.result() for task in tasks]

    async def run_task(
        self, position: Position, index: int, node_input: object, run: Run, step_exits: AsyncExitStack
    ) -> TaskOutput | Interrupted:
        address = TaskAddress(checkpoint_id=position.checkpoint_id, index=index)
        node_name = position.tasks[index].node
        recorded = position.progress.get(index)
        match recorded:
            case TaskOutput():
                return recorded
            case InterruptedTask() if address not in run.answers:
                return interrupt_task(address, node_name, recorded)
            case InterruptedTask() | None:
                return await self.execute_task(address, node_name, node_input, run, step_exits)
            case _:
                assert_never(recorded)

    async def execute_task(
        self, address: TaskAddress, node_name: str, node_input: object, run: Run, step_exits: AsyncExitStack
    ) -> TaskOutput | Interrupted:
        node = self.nodes[node_name]
        match node.action:
            case CallPlan() as plan:
                answers = run.answers.get(address, ())
                current_interrupts.set(TaskInterrupts(answers))
                result = await self.call_function(plan, node, node_input, run, step_exits)
                match result:
                    case NodeInterrupt():
                        state = InterruptedTask(answers=answers, payload=result.payload)
                        await finish_despite_cancellation(run.journal.record_interrupt(address, state))
                        return interrupt_task(address, node_name, state)
                    case TaskOutput():
                        output = result
                    case _:
                        assert_never(result)
            case CompiledGraph() as child:
                start = await child.enter(run.journal, address, node_input)
                child_result = await child.run_steps(start, run)
                match child_result:
                    case Finished(output=child_output):
                        output = self.read_task_output(child_output, node)
                    case Interrupted():
                        return child_result
                    case _:
                        assert_never(child_result)
            case _:
                assert_never(node.action)
        return await finish_despite_cancellation(run.journal.record_output(self, address, output))

    def read_task_output(self, result: object, node: CompiledNode) -> TaskOutput:
        outcome = read_outcome(result, node.targets, self.nodes.keys())
        pushes = tuple(PushTask(send.node, send.arg) for send in outcome.routes.sends)
        return TaskOutput(outcome.writes, outcome.routes.targets, pushes)

    async def call_function(
        self, plan: CallPlan, node: CompiledNode, node_input: object, run: Run, step_exits: AsyncExitStack
    ) -> TaskOutput | NodeInterrupt:
        if plan.run_providers:
            async with run.lock:
                await open_providers(plan.run_providers, run.values, run.exits)
        node_values = ChainMap[Depends, object]({}, dict(run.values))
        output: TaskOutput | None = None
        try:
            async with AsyncExitStack[bool]() as node_exits:
                result = await execute_plan(plan, node_input, node_values, node_exits)
                output = self.read_task_output(result, node)
                step_exits.push_async_callback(node_exits.pop_all().aclose)
        except NodeInterrupt as interruption:
            return interruption
        except BaseExceptionGroup as group:
            return find_interruption(group)
        if output is None:
            raise RuntimeError(f"a dependency of {plan.function!r} suppressed the exception of its node")
        return output


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
        input_schema: type[object] | None = None,
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
        input_schemas: dict[str, type[object]] = {
            **{name: self.node_input_schemas.get(name, self.state_schema) for name in self.nodes},
            **{name: child.input_schema for name, child in self.child_graphs.items()},
        }
        actions: dict[str, CallPlan | CompiledGraph[object, Never, object]] = {
            **{name: plan_call(node, overrides) for name, node in self.nodes.items()},
            **{name: child.compile(overrides) for name, child in self.child_graphs.items()},
        }
        return CompiledGraph(
            state_schema=self.state_schema,
            state_fields=frozenset(build_channels(self.state_schema)),
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

import asyncio
import inspect
from collections import ChainMap
from collections.abc import AsyncGenerator, Callable, Collection, Generator, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Never, Protocol, assert_never, overload
from uuid import NAMESPACE_OID, uuid5

from pydantic import JsonValue, TypeAdapter

from pregel_redux.channels import Channel, extract_writes, parse_updates
from pregel_redux.checkpoints import CheckpointSource
from pregel_redux.commands import Endpoint, Send, read_outcome
from pregel_redux.dependencies import (
    CallPlan,
    Depends,
    ShieldedExitStack,
    close_at_exit,
    execute_plan,
    finish_despite_cancellation,
    open_providers,
)
from pregel_redux.graph.adapters import iterate_in_new_loop, run_in_new_loop, stream_run
from pregel_redux.graph.task_refs import NodeCall, TaskRef, name_task, refer_task
from pregel_redux.interrupts import (
    Finished,
    GraphInterrupt,
    Interrupt,
    Interrupted,
    ResumeInput,
    Scratchpad,
    StopReason,
    current_scratchpad,
)
from pregel_redux.pregel import (
    Arrival,
    BranchChannel,
    EphemeralValue,
    JoinChannel,
    JoinKey,
    LastValueAfterFinish,
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
    Position,
    PullTask,
    PushTask,
    StartTask,
    Task,
    TaskWrites,
    TriggerKey,
    apply_writes,
    open_channels,
    plan_tasks,
)
from pregel_redux.settings import RunSettings
from pregel_redux.shutdown import ShutdownController, ShutdownSignal, run_until_abort
from pregel_redux.snapshots import StateSnapshot
from pregel_redux.streams import (
    CheckpointPart,
    CustomPart,
    InterruptsPart,
    Stream,
    StreamMode,
    StreamPart,
    TaskResultPart,
    TaskStartPart,
    UpdatesPart,
    ValuesPart,
    current_stream_writer,
)


@dataclass(frozen=True)
class ChildEntry:
    position: Position
    journal: "Journal"
    resumed: bool


class Journal(Protocol):
    @property
    def ns(self) -> tuple[str, ...]: ...
    async def save(self, position: Position, source: CheckpointSource) -> None: ...
    async def record(self, task: TaskRef, writes: TaskWrites) -> None: ...
    async def record_waiting(self, task: TaskRef, interrupts: Sequence[Interrupt]) -> None: ...
    async def record_failure(self, task: TaskRef, error: str) -> None: ...
    async def record_answers(self, task: TaskRef, answers: Sequence[JsonValue]) -> None: ...
    def find_writes(self, task: TaskRef) -> TaskWrites | None: ...
    def make_task_id(self, task: TaskRef) -> str: ...
    async def read_snapshot(self) -> StateSnapshot | None: ...
    def open_scratchpad(self, task: TaskRef, resume: ResumeInput) -> Scratchpad: ...
    async def enter_child(self, child: "CompiledGraph[object, Never, object]", call: NodeCall) -> ChildEntry: ...


@dataclass
class NoJournal:
    ns: tuple[str, ...]
    step: int

    async def save(self, position: Position, source: CheckpointSource) -> None:
        self.step += 1

    def make_task_id(self, task: TaskRef) -> str:
        return str(uuid5(NAMESPACE_OID, f"{'|'.join(self.ns)}|{self.step}|{name_task(task)}"))

    async def read_snapshot(self) -> None:
        return None

    async def record(self, task: TaskRef, writes: TaskWrites) -> None:
        pass

    async def record_waiting(self, task: TaskRef, interrupts: Sequence[Interrupt]) -> None:
        pass

    async def record_failure(self, task: TaskRef, error: str) -> None:
        pass

    async def record_answers(self, task: TaskRef, answers: Sequence[JsonValue]) -> None:
        pass

    def find_writes(self, task: TaskRef) -> TaskWrites | None:
        return None

    def open_scratchpad(self, task: TaskRef, resume: ResumeInput) -> Scratchpad:
        return Scratchpad(str(task.node), [], resume, 0)

    async def enter_child(self, child: "CompiledGraph[object, Never, object]", call: NodeCall) -> ChildEntry:
        start = child.feed(child.open(extract_writes(call.node_input)), {})
        child_ns = (*self.ns, f"{call.task.node}:{self.make_task_id(call.task)}")
        return ChildEntry(start, NoJournal(child_ns, 0), resumed=False)


@dataclass(frozen=True)
class Run:
    values: dict[Depends, object]
    exits: AsyncExitStack
    lock: asyncio.Lock
    step_limit: int
    resume: ResumeInput
    stream: Stream
    shutdown: ShutdownSignal


@dataclass(frozen=True)
class CompiledNode:
    action: "CallPlan | CompiledGraph[object, Never, object]"
    input_schema: type[object]
    input_fields: frozenset[str]
    targets: frozenset[str]
    joins: frozenset[JoinKey]
    triggers: tuple[TriggerKey, ...]
    defer: bool

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
class TaskStopped:
    interrupts: tuple[Interrupt, ...]
    reason: StopReason


@dataclass(frozen=True)
class CompiledGraph[StateT, InputT, OutputT]:
    output_schema: Callable[..., OutputT]
    channels: Mapping[str, Channel]
    output_fields: frozenset[str]
    nodes: Mapping[str, CompiledNode]
    entry_nodes: frozenset[str]
    interrupt_before: frozenset[str]
    interrupt_after: frozenset[str]

    async def ainvoke(self, input: InputT, *, step_limit: int | None = None) -> OutputT:
        return await self.run_without_thread(
            input, step_limit, Stream(frozenset(), subgraphs=False, emit=lambda part: None)
        )

    async def run_without_thread(self, input: InputT, step_limit: int | None, stream: Stream) -> OutputT:
        start = self.feed(self.open({}), extract_writes(input))
        result = await self.run(
            start,
            step_limit,
            NoJournal((), 0),
            ResumeInput([], {}),
            stream,
            {},
            ShutdownController().signal,
            resumed=False,
        )
        match result:
            case Finished():
                return result.output
            case Interrupted():
                raise RuntimeError(
                    f"the run stopped at {result.interrupts!r}, and only a graph with a checkpointer can continue it"
                )
            case _:
                assert_never(result)

    async def run(
        self,
        position: Position,
        step_limit: int | None,
        journal: Journal,
        resume: ResumeInput,
        stream: Stream,
        provided: dict[Depends, object],
        shutdown: ShutdownSignal,
        *,
        resumed: bool,
    ) -> Finished[OutputT] | Interrupted:
        limit = RunSettings().step_limit if step_limit is None else step_limit
        async with ShieldedExitStack() as run_exits:
            run = Run(provided, run_exits, asyncio.Lock(), limit, resume, stream, shutdown)
            return await run_until_abort(shutdown, self.run_steps(position, run, journal, resumed=resumed))
        raise RuntimeError("a run-scoped dependency suppressed the exception of its run")

    @overload
    def astream(
        self,
        input: InputT,
        *,
        stream_mode: Literal[StreamMode.values],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> AsyncGenerator[ValuesPart]: ...
    @overload
    def astream(
        self,
        input: InputT,
        *,
        stream_mode: Literal[StreamMode.updates],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> AsyncGenerator[UpdatesPart | InterruptsPart]: ...
    @overload
    def astream(
        self,
        input: InputT,
        *,
        stream_mode: Literal[StreamMode.custom],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> AsyncGenerator[CustomPart]: ...
    @overload
    def astream(
        self,
        input: InputT,
        *,
        stream_mode: Literal[StreamMode.tasks],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> AsyncGenerator[TaskStartPart | TaskResultPart]: ...
    @overload
    def astream(
        self,
        input: InputT,
        *,
        stream_mode: Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> AsyncGenerator[StreamPart]: ...
    def astream(
        self,
        input: InputT,
        *,
        stream_mode: StreamMode | Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> AsyncGenerator[StreamPart]:
        modes = frozenset([stream_mode] if isinstance(stream_mode, StreamMode) else stream_mode)
        return stream_run(
            lambda emit: self.run_without_thread(input, step_limit, Stream(modes, subgraphs=subgraphs, emit=emit))
        )

    @overload
    def stream(
        self,
        input: InputT,
        *,
        stream_mode: Literal[StreamMode.values],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> Generator[ValuesPart]: ...
    @overload
    def stream(
        self,
        input: InputT,
        *,
        stream_mode: Literal[StreamMode.updates],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> Generator[UpdatesPart | InterruptsPart]: ...
    @overload
    def stream(
        self,
        input: InputT,
        *,
        stream_mode: Literal[StreamMode.custom],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> Generator[CustomPart]: ...
    @overload
    def stream(
        self,
        input: InputT,
        *,
        stream_mode: Literal[StreamMode.tasks],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> Generator[TaskStartPart | TaskResultPart]: ...
    @overload
    def stream(
        self,
        input: InputT,
        *,
        stream_mode: Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> Generator[StreamPart]: ...
    def stream(
        self,
        input: InputT,
        *,
        stream_mode: StreamMode | Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
    ) -> Generator[StreamPart]:
        self.reject_async_nodes("astream")
        modes = frozenset([stream_mode] if isinstance(stream_mode, StreamMode) else stream_mode)
        return iterate_in_new_loop(
            lambda: self.astream(input, stream_mode=modes, subgraphs=subgraphs, step_limit=step_limit)
        )

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

    def invoke(self, input: InputT, *, step_limit: int | None = None) -> OutputT:
        self.reject_async_nodes("ainvoke")
        return run_in_new_loop(lambda: self.ainvoke(input, step_limit=step_limit))

    @cached_property
    def triggers(self) -> Mapping[str, tuple[TriggerKey, ...]]:
        return {name: node.triggers for name, node in self.nodes.items()}

    @cached_property
    def trigger_keys(self) -> frozenset[TriggerKey]:
        return frozenset(key for keys in self.triggers.values() for key in keys)

    def open(self, given: Mapping[str, object]) -> Position:
        branches: dict[str, BranchChannel] = {
            name: LastValueAfterFinish(None, finished=False) if node.defer else EphemeralValue(None)
            for name, node in self.nodes.items()
        }
        joins: dict[JoinKey, JoinChannel] = {
            key: (
                NamedBarrierValueAfterFinish(key.sources, frozenset(), finished=False)
                if self.nodes[key.target].defer
                else NamedBarrierValue(key.sources, frozenset())
            )
            for node in self.nodes.values()
            for key in node.joins
        }
        return open_channels(self.channels, given, branches, joins)

    def feed(self, position: Position, input_writes: Mapping[str, object]) -> Position:
        discarded = plan_tasks(position, self.triggers)
        return apply_writes(position, discarded, [], [parse_updates(self.channels, input_writes)], self.trigger_keys)

    def build_call(self, task: Task, position: Position) -> NodeCall:
        ref = refer_task(task)
        match task:
            case StartTask():
                return NodeCall(ref, task.writes)
            case PullTask():
                return NodeCall(ref, self.nodes[task.node].build_input(position.channels.read_values()))
            case PushTask():
                return NodeCall(ref, self.nodes[task.send.node].build_send_input(task.send))
            case _:
                assert_never(task)

    async def run_steps(
        self, start: Position, run: Run, journal: Journal, *, resumed: bool
    ) -> Finished[OutputT] | Interrupted:
        position = start
        steps = 0
        skip_interrupt_before = resumed
        if not resumed:
            await self.send_checkpoint(run, journal)
        while tasks := plan_tasks(position, self.triggers):
            calls = [self.build_call(task, position) for task in tasks]
            names = {call.task.node for call in calls}
            if not skip_interrupt_before and not names.isdisjoint(self.interrupt_before):
                return self.stop(position, run, journal, (), StopReason.breakpoint)
            if run.shutdown.requested and names != {Endpoint.start}:
                snapshot = await journal.read_snapshot()
                waiting = () if snapshot is None else snapshot.interrupts
                return self.stop(position, run, journal, waiting, StopReason.shutdown)
            skip_interrupt_before = False
            if names != {Endpoint.start}:
                if steps == run.step_limit:
                    raise RuntimeError(f"the graph did not finish within {run.step_limit} steps")
                steps += 1
            async with ShieldedExitStack() as step_exits:
                results = await self.run_step(calls, run, journal, step_exits)
                stops = [result for result in results if isinstance(result, TaskStopped)]
                if stops:
                    interrupts = tuple(item for stopped in stops for item in stopped.interrupts)
                    by_shutdown = all(stopped.reason is StopReason.shutdown for stopped in stops)
                    reason = StopReason.shutdown if by_shutdown else StopReason.question
                    return self.stop(position, run, journal, interrupts, reason)
                writes = [result for result in results if isinstance(result, TaskWrites)]
                position = apply_writes(position, tasks, writes, [], self.trigger_keys)
                await journal.save(position, CheckpointSource.loop)
                await self.send_checkpoint(run, journal)
            await asyncio.sleep(0)  # deliver a pending cancellation
            run.stream.send(StreamMode.values, ValuesPart(journal.ns, position.channels.read_values(), ()))
            if not names.isdisjoint(self.interrupt_after):
                return self.stop(position, run, journal, (), StopReason.breakpoint)
        values = position.channels.read_values()
        return Finished(self.output_schema(**{name: values[name] for name in self.output_fields if name in values}))

    async def send_checkpoint(self, run: Run, journal: Journal) -> None:
        if StreamMode.checkpoints in run.stream.modes:
            snapshot = await journal.read_snapshot()
            if snapshot is not None:
                run.stream.send(StreamMode.checkpoints, CheckpointPart(journal.ns, snapshot))

    def stop(
        self, position: Position, run: Run, journal: Journal, interrupts: tuple[Interrupt, ...], reason: StopReason
    ) -> Interrupted:
        values = position.channels.read_values()
        run.stream.send(StreamMode.values, ValuesPart(journal.ns, values, interrupts))
        run.stream.send(StreamMode.updates, InterruptsPart(journal.ns, interrupts))
        return Interrupted(interrupts, values, reason)

    async def run_step(
        self, calls: Sequence[NodeCall], run: Run, journal: Journal, step_exits: AsyncExitStack
    ) -> list[TaskWrites | TaskStopped]:
        tasks = [asyncio.create_task(self.run_task(call, run, journal, step_exits)) for call in calls]
        try:
            await asyncio.wait(tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await finish_despite_cancellation(asyncio.wait(tasks))
            raise
        failures = [failure for task in tasks if (failure := task.exception()) is not None]
        match failures:
            case []:
                return [task.result() for task in tasks]
            case [failure]:
                raise failure
            case _:
                raise BaseExceptionGroup("several nodes of one step failed", failures)

    async def run_task(
        self, call: NodeCall, run: Run, journal: Journal, step_exits: AsyncExitStack
    ) -> TaskWrites | TaskStopped:
        saved = journal.find_writes(call.task)
        if saved is not None:
            return saved
        scratchpad = journal.open_scratchpad(call.task, run.resume)
        current_scratchpad.set(scratchpad)
        current_stream_writer.set(lambda data: run.stream.send(StreamMode.custom, CustomPart(journal.ns, data)))
        name = call.task.node
        task_id = journal.make_task_id(call.task)
        if isinstance(name, str):
            run.stream.send(StreamMode.tasks, TaskStartPart(journal.ns, task_id, name, call.node_input))
        saved_answers = len(scratchpad.resume)
        try:
            outcome = await self.run_node(call, run, journal, step_exits)
        except GraphInterrupt as stop:
            outcome = Interrupted(stop.interrupts, {}, StopReason.question)
        except Exception as error:
            await journal.record_failure(call.task, repr(error))
            raise
        finally:
            if len(scratchpad.resume) > saved_answers:
                await journal.record_answers(call.task, scratchpad.resume)
        match outcome:
            case TaskWrites():
                await journal.record(call.task, outcome)
                if isinstance(name, str):
                    run.stream.send(StreamMode.updates, UpdatesPart(journal.ns, name, outcome.fields))
                    run.stream.send(StreamMode.tasks, TaskResultPart(journal.ns, task_id, name, outcome.fields, ()))
                return outcome
            case Interrupted(reason=StopReason.shutdown):
                return TaskStopped(outcome.interrupts, outcome.reason)
            case Interrupted():
                await journal.record_waiting(call.task, outcome.interrupts)
                part = TaskResultPart(journal.ns, task_id, str(name), {}, outcome.interrupts)
                run.stream.send(StreamMode.tasks, part)
                return TaskStopped(outcome.interrupts, outcome.reason)
            case _:
                assert_never(outcome)

    async def run_node(
        self, call: NodeCall, run: Run, journal: Journal, step_exits: AsyncExitStack
    ) -> TaskWrites | Interrupted:
        match call.task.node:
            case Endpoint.start:
                return TaskWrites(
                    parse_updates(self.channels, extract_writes(call.node_input)), self.entry_nodes, (), ()
                )
            case str() as name:
                action = self.nodes[name].action
                match action:
                    case CallPlan():
                        return await self.call_function(action, name, call, run, step_exits)
                    case CompiledGraph():
                        entry = await journal.enter_child(action, call)
                        result = await action.run_steps(entry.position, run, entry.journal, resumed=entry.resumed)
                        match result:
                            case Finished():
                                return self.read_writes(name, result.output)
                            case Interrupted():
                                return result
                            case _:
                                assert_never(result)
                    case _:
                        assert_never(action)
            case _:
                assert_never(call.task.node)

    def read_writes(self, name: str, result: object) -> TaskWrites:
        node = self.nodes[name]
        outcome = read_outcome(result, node.targets, self.nodes.keys())
        return TaskWrites(
            fields=parse_updates(self.channels, outcome.writes),
            branches=outcome.routes.targets,
            arrivals=tuple(Arrival(key, name) for key in node.joins),
            sends=outcome.routes.sends,
        )

    async def call_function(
        self, plan: CallPlan, name: str, call: NodeCall, run: Run, step_exits: AsyncExitStack
    ) -> TaskWrites:
        if plan.run_providers:
            async with run.lock:
                await open_providers(plan.run_providers, run.values, run.exits)
        node_values = ChainMap[Depends, object]({}, dict(run.values))
        writes: TaskWrites | None = None
        async with AsyncExitStack[bool]() as node_exits:
            result = await execute_plan(plan, call.node_input, node_values, node_exits)
            writes = self.read_writes(name, result)
            close_at_exit(node_exits.pop_all(), step_exits)
        if writes is None:
            raise RuntimeError(f"a dependency of {plan.function!r} suppressed the exception of its node")
        return writes

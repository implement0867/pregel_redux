import asyncio
import inspect
from collections import ChainMap
from collections.abc import AsyncGenerator, Callable, Generator, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import cached_property
from typing import Never, Protocol, assert_never
from uuid import NAMESPACE_OID, uuid5

from pydantic import TypeAdapter
from typing_extensions import TypeForm

from pregel_redux.channels import Channel, extract_writes, parse_updates
from pregel_redux.checkpoints import CheckpointSource
from pregel_redux.commands import Send, read_outcome
from pregel_redux.configs import Namespace, Segment, encode_namespace
from pregel_redux.dependencies import (
    CallPlan,
    Request,
    ShieldedExitStack,
    close_at_exit,
    execute_plan,
    finish_despite_cancellation,
    open_providers,
)
from pregel_redux.graph.adapters import iterate_in_new_loop, run_in_new_loop, stream_run
from pregel_redux.messages import Entrance, MessageSchema
from pregel_redux.pregel import (
    Arrival,
    BranchChannel,
    EphemeralValue,
    HeldValue,
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
    find_waiting,
    open_channels,
    plan_tasks,
)
from pregel_redux.results import Finished, Stopped, StopReason, Waiting
from pregel_redux.settings import RunSettings
from pregel_redux.shutdown import ShutdownController, ShutdownSignal, run_until_abort
from pregel_redux.streams import (
    CheckpointPart,
    ChildPart,
    StoppedPart,
    Stream,
    StreamPart,
    StreamWriter,
    TaskResultPart,
    TaskStartPart,
    UpdatesPart,
    ValuesPart,
    WaitingPart,
)
from pregel_redux.task_refs import NodeCall, PullRef, PushRef, StartRef, TaskRef, name_task, refer_task


@dataclass(frozen=True)
class FreshStart:
    position: Position
    journal: "Journal"


@dataclass(frozen=True)
class Continuation:
    position: Position
    journal: "Journal"


type Start = FreshStart | Continuation


@dataclass(frozen=True)
class Breakpoints:
    before: frozenset[str]
    after: frozenset[str]


@dataclass(frozen=True)
class RunOptions:
    step_limit: int | None
    stream: Stream
    shutdown: ShutdownSignal
    provided: Mapping[Request, object]
    breakpoints: Breakpoints


class Journal(Protocol):
    @property
    def ns(self) -> Namespace: ...
    async def save(self, position: Position, source: CheckpointSource) -> None: ...
    async def record(self, task: TaskRef, writes: TaskWrites) -> None: ...
    async def record_failure(self, task: TaskRef, error: str) -> None: ...
    async def record_waiting(self, task: TaskRef) -> None: ...
    def find_writes(self, task: TaskRef) -> TaskWrites | None: ...
    def make_task_id(self, task: TaskRef) -> str: ...
    async def read_checkpoint(self) -> CheckpointPart | None: ...
    async def enter_child(self, child: "CompiledGraph[object, Never, object]", call: NodeCall) -> Start: ...


@dataclass
class NoJournal:
    ns: Namespace
    step: int

    async def save(self, position: Position, source: CheckpointSource) -> None:
        self.step += 1

    def make_task_id(self, task: TaskRef) -> str:
        return str(uuid5(NAMESPACE_OID, f"{encode_namespace(self.ns)}|{self.step}|{name_task(task)}"))

    async def read_checkpoint(self) -> None:
        return None

    async def record(self, task: TaskRef, writes: TaskWrites) -> None:
        pass

    async def record_failure(self, task: TaskRef, error: str) -> None:
        pass

    async def record_waiting(self, task: TaskRef) -> None:
        pass

    def find_writes(self, task: TaskRef) -> TaskWrites | None:
        return None

    async def enter_child(self, child: "CompiledGraph[object, Never, object]", call: NodeCall) -> Start:
        start = child.feed(child.open(extract_writes(call.node_input)), {})
        child_ns = (*self.ns, Segment(call.task.name, self.make_task_id(call.task)))
        return FreshStart(start, NoJournal(child_ns, 0))


@dataclass(frozen=True)
class Run:
    values: dict[Request, object]
    exits: AsyncExitStack
    lock: asyncio.Lock
    step_limit: int
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
    message: MessageSchema | None

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
class CompiledGraph[StateT, InputT, OutputT]:
    """Runs a graph in memory from its input to its output."""

    output_schema: Callable[..., OutputT]
    channels: Mapping[str, Channel]
    output_fields: frozenset[str]
    nodes: Mapping[str, CompiledNode]
    entry_nodes: frozenset[str]

    async def ainvoke(self, input: InputT, *, step_limit: int | None = None) -> OutputT:
        """Runs the graph on `input` and returns the output.

        Parameters
        ----------
        step_limit
            The `PREGEL_REDUX_STEP_LIMIT` environment variable sets the default.
            The built-in default is 25.
        """
        return await self.run_without_thread(input, step_limit, Stream(lambda part: None))

    async def run_without_thread(self, input: InputT, step_limit: int | None, stream: Stream) -> OutputT:
        start = FreshStart(self.feed(self.open({}), extract_writes(input)), NoJournal((), 0))
        options = RunOptions(
            step_limit=step_limit,
            stream=stream,
            shutdown=ShutdownController().signal,
            provided={},
            breakpoints=Breakpoints(frozenset(), frozenset()),
        )
        result = await self.run(start, options)
        match result:
            case Finished():
                return result.output
            case Waiting() | Stopped():
                raise RuntimeError(f"the run ended as {result!r}, and only a graph with a checkpointer can continue it")
            case _:
                assert_never(result)

    async def run(self, start: Start, options: RunOptions) -> Finished[OutputT] | Waiting | Stopped:
        limit = RunSettings().step_limit if options.step_limit is None else options.step_limit
        async with ShieldedExitStack() as run_exits:
            run = Run(
                dict(options.provided),
                run_exits,
                asyncio.Lock(),
                limit,
                options.stream,
                options.shutdown,
            )
            return await run_until_abort(options.shutdown, self.run_steps(start, run, options.breakpoints))
        raise RuntimeError("a run-scoped dependency suppressed the exception of its run")

    def astream(
        self,
        input: InputT,
        *,
        step_limit: int | None = None,
    ) -> AsyncGenerator[StreamPart | ChildPart]:
        """Runs the graph on `input` and yields each part that the run emits.

        The parameters match `ainvoke`.
        """
        return stream_run(lambda emit: self.run_without_thread(input, step_limit, Stream(emit)))

    def stream(
        self,
        input: InputT,
        *,
        step_limit: int | None = None,
    ) -> Generator[StreamPart | ChildPart]:
        """Runs the graph on `input` and yields each part that the run emits.

        Each iteration blocks until the run emits the next part.
        A coroutine node function raises `TypeError`.
        """
        self.reject_async_nodes("astream")
        return iterate_in_new_loop(lambda: self.astream(input, step_limit=step_limit))

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
        """Runs the graph on `input` and returns the output.

        The call blocks until the run ends.
        A coroutine node function raises `TypeError`.
        """
        self.reject_async_nodes("ainvoke")
        return run_in_new_loop(lambda: self.ainvoke(input, step_limit=step_limit))

    @cached_property
    def receivers(self) -> Mapping[str, MessageSchema]:
        return {name: node.message for name, node in self.nodes.items() if node.message is not None}

    def find_receivers(self) -> list[str]:
        names = list(self.receivers)
        for name, node in self.nodes.items():
            match node.action:
                case CallPlan():
                    pass
                case CompiledGraph() as child:
                    names.extend(f"{name}.{child_name}" for child_name in child.find_receivers())
                case _:
                    assert_never(node.action)
        return sorted(names)

    def list_entrances(self, ns: Namespace, position: Position) -> dict[Entrance, TypeForm[object]]:
        return {Entrance(ns, node): self.receivers[node].message_type for node in find_waiting(position, self.triggers)}

    @cached_property
    def triggers(self) -> Mapping[str, tuple[TriggerKey, ...]]:
        return {name: node.triggers for name, node in self.nodes.items()}

    @cached_property
    def trigger_keys(self) -> frozenset[TriggerKey]:
        return frozenset(key for keys in self.triggers.values() for key in keys)

    def open(self, given: Mapping[str, object]) -> Position:
        branches: dict[str, BranchChannel] = {
            name: (
                HeldValue(None)
                if name in self.receivers
                else LastValueAfterFinish(None, finished=False)
                if node.defer
                else EphemeralValue(None)
            )
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
        return open_channels(self.channels, given, branches, joins, self.receivers)

    def feed(self, position: Position, input_writes: Mapping[str, object]) -> Position:
        discarded = [
            *plan_tasks(position, self.triggers),
            *(PullTask(node, self.triggers[node]) for node in find_waiting(position, self.triggers)),
        ]
        return apply_writes(position, discarded, [], [parse_updates(self.channels, input_writes)], self.trigger_keys)

    def build_call(self, task: Task, position: Position) -> NodeCall:
        ref = refer_task(task)
        match task:
            case StartTask():
                return NodeCall(ref, task.writes, None)
            case PullTask():
                node_input = self.nodes[task.node].build_input(position.channels.read_values())
                return NodeCall(ref, node_input, position.channels.received.get(task.node))
            case PushTask():
                return NodeCall(ref, self.nodes[task.send.node].build_send_input(task.send), None)
            case _:
                assert_never(task)

    async def run_steps(
        self, start: Start, run: Run, breakpoints: Breakpoints
    ) -> Finished[OutputT] | Waiting | Stopped:
        position, journal = start.position, start.journal
        steps = 0
        match start:
            case FreshStart():
                skip_interrupt_before = False
                await self.send_checkpoint(run, journal)
            case Continuation():
                skip_interrupt_before = True
            case _:
                assert_never(start)
        while tasks := plan_tasks(position, self.triggers):
            calls = [self.build_call(task, position) for task in tasks]
            names = {call.task.name for call in calls}
            runs_nodes = any(not isinstance(task, StartTask) for task in tasks)
            if not skip_interrupt_before and not names.isdisjoint(breakpoints.before):
                return self.stop(position, run, journal, StopReason.breakpoint)
            if run.shutdown.requested and runs_nodes:
                return self.stop(position, run, journal, StopReason.shutdown)
            skip_interrupt_before = False
            if runs_nodes:
                if steps == run.step_limit:
                    raise RuntimeError(f"the graph did not finish within {run.step_limit} steps")
                steps += 1
            async with ShieldedExitStack() as step_exits:
                results = await self.run_step(calls, run, journal, step_exits)
                if any(isinstance(result, Stopped) for result in results):
                    return self.stop(position, run, journal, StopReason.shutdown)
                accepts = {
                    entrance: message_type
                    for result in results
                    if isinstance(result, Waiting)
                    for entrance, message_type in result.accepts.items()
                }
                if accepts:
                    return self.wait(position, run, journal, accepts)
                writes = [result for result in results if isinstance(result, TaskWrites)]
                position = apply_writes(position, tasks, writes, [], self.trigger_keys)
                await journal.save(position, CheckpointSource.loop)
                await self.send_checkpoint(run, journal)
            await asyncio.sleep(0)  # deliver a pending cancellation
            run.stream.send(journal.ns, ValuesPart(position.channels.read_values()))
            if not names.isdisjoint(breakpoints.after):
                return self.stop(position, run, journal, StopReason.breakpoint)
        accepts = self.list_entrances(journal.ns, position)
        if accepts:
            return self.wait(position, run, journal, accepts)
        values = position.channels.read_values()
        return Finished(self.output_schema(**{name: values[name] for name in self.output_fields if name in values}))

    async def send_checkpoint(self, run: Run, journal: Journal) -> None:
        part = await journal.read_checkpoint()
        if part is not None:
            run.stream.send(journal.ns, part)

    def stop(self, position: Position, run: Run, journal: Journal, reason: StopReason) -> Stopped:
        values = position.channels.read_values()
        run.stream.send(journal.ns, ValuesPart(values))
        run.stream.send(journal.ns, StoppedPart(reason))
        return Stopped(values, reason)

    def wait(
        self, position: Position, run: Run, journal: Journal, accepts: Mapping[Entrance, TypeForm[object]]
    ) -> Waiting:
        run.stream.send(journal.ns, WaitingPart(accepts))
        return Waiting(position.channels.read_values(), accepts)

    async def run_step(
        self, calls: Sequence[NodeCall], run: Run, journal: Journal, step_exits: AsyncExitStack
    ) -> list[TaskWrites | Waiting | Stopped]:
        tasks = [asyncio.create_task(self.run_task(call, run, journal, step_exits)) for call in calls]
        try:
            await asyncio.wait(tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await finish_despite_cancellation(asyncio.gather(*tasks, return_exceptions=True))
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
    ) -> TaskWrites | Waiting | Stopped:
        saved = journal.find_writes(call.task)
        if saved is not None:
            return saved
        name = call.task.name
        task_id = journal.make_task_id(call.task)
        match call.task:
            case StartRef():
                is_node = False
            case PullRef() | PushRef():
                is_node = True
            case _:
                assert_never(call.task)
        if is_node:
            run.stream.send(journal.ns, TaskStartPart(task_id, name, call.node_input))
        try:
            outcome = await self.run_node(call, run, journal, step_exits)
        except Exception as error:
            await journal.record_failure(call.task, repr(error))
            raise
        match outcome:
            case TaskWrites():
                await journal.record(call.task, outcome)
                if is_node:
                    run.stream.send(journal.ns, UpdatesPart(name, outcome.fields))
                    run.stream.send(journal.ns, TaskResultPart(task_id, name, outcome.fields))
                return outcome
            case Waiting():
                await journal.record_waiting(call.task)
                return outcome
            case Stopped():
                return outcome
            case _:
                assert_never(outcome)

    async def run_node(
        self, call: NodeCall, run: Run, journal: Journal, step_exits: AsyncExitStack
    ) -> TaskWrites | Waiting | Stopped:
        match call.task:
            case StartRef():
                return TaskWrites(
                    parse_updates(self.channels, extract_writes(call.node_input)), self.entry_nodes, (), ()
                )
            case PullRef(name=name) | PushRef(name=name):
                action = self.nodes[name].action
                match action:
                    case CallPlan():
                        return await self.call_function(action, name, call, run, journal.ns, step_exits)
                    case CompiledGraph():
                        entry = await journal.enter_child(action, call)
                        result = await action.run_steps(entry, run, Breakpoints(frozenset(), frozenset()))
                        match result:
                            case Finished():
                                return self.read_writes(name, result.output)
                            case Waiting() | Stopped():
                                return result
                            case _:
                                assert_never(result)
                    case _:
                        assert_never(action)
            case _:
                assert_never(call.task)

    def read_writes(self, name: str, result: object) -> TaskWrites:
        node = self.nodes[name]
        outcome = read_outcome(result, node.targets, self.nodes.keys())
        self.reject_sends_to_receivers(outcome.routes.sends)
        return TaskWrites(
            fields=parse_updates(self.channels, outcome.writes),
            branches=outcome.routes.targets,
            arrivals=tuple(Arrival(key, name) for key in node.joins),
            sends=outcome.routes.sends,
        )

    def reject_sends_to_receivers(self, sends: Sequence[Send]) -> None:
        waiting = sorted({send.node for send in sends} & self.receivers.keys())
        if waiting:
            raise ValueError(
                f"a Send starts its node in the next step, and the nodes {waiting!r} wait for a message from outside"
            )

    async def call_function(
        self, plan: CallPlan, name: str, call: NodeCall, run: Run, ns: Namespace, step_exits: AsyncExitStack
    ) -> TaskWrites:
        if plan.run_providers:
            async with run.lock:
                await open_providers(plan.run_providers, run.values, run.exits)
        node_values = ChainMap[Request, object]({StreamWriter: StreamWriter(run.stream, ns)}, dict(run.values))
        writes: TaskWrites | None = None
        async with AsyncExitStack[bool]() as node_exits:
            result = await execute_plan(plan, call.node_input, call.received, node_values, node_exits)
            writes = self.read_writes(name, result)
            close_at_exit(node_exits.pop_all(), step_exits)
        if writes is None:
            raise RuntimeError(f"a dependency of {plan.function!r} suppressed the exception of its node")
        return writes

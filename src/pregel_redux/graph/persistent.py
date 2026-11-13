import secrets
import time
from collections.abc import AsyncGenerator, AsyncIterator, Collection, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, Never, assert_never, overload
from uuid import UUID, uuid5

from pydantic import JsonValue

from pregel_redux.channels import (
    Stored,
    dump_json,
    extract_writes,
    load_json,
    parse_updates,
    reject_unknown_channels,
)
from pregel_redux.checkpoints import (
    AsyncCheckpointer,
    ChannelValues,
    ChannelVersions,
    Checkpoint,
    Checkpointer,
    CheckpointSource,
    CheckpointTuple,
    JoinVersion,
    PendingSend,
    PendingWrite,
    RunCommandRecord,
    SavedArrival,
    SavedBranch,
    SavedJoin,
    SavedWrites,
    TaskAnswersRecord,
    TaskFailed,
    TaskStopRecord,
    TaskWaiting,
    TaskWritesRecord,
    ThreadedCheckpointer,
    TriggerVersions,
)
from pregel_redux.commands import Command, Endpoint, Send, resolve_goto
from pregel_redux.configs import ThreadConfig, parse_run_target, parse_state_address, read_config
from pregel_redux.dependencies import Depends, Scope, finish_despite_cancellation
from pregel_redux.graph.compiled import (
    ChildEntry,
    CompiledGraph,
    NodeCall,
    TaskRef,
    iterate_in_new_loop,
    name_task,
    refer_task,
    run_in_new_loop,
    stream_run,
)
from pregel_redux.interrupts import Finished, Interrupt, Interrupted, ResumeInput, Scratchpad, make_interrupt_id
from pregel_redux.pregel import (
    Arrival,
    BranchChannel,
    BranchKey,
    ChannelKey,
    Channels,
    EphemeralValue,
    FieldKey,
    JoinChannel,
    JoinKey,
    LastValueAfterFinish,
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
    NodeName,
    Position,
    PullTask,
    StartKey,
    Task,
    TasksKey,
    TaskWrites,
    Topic,
    TriggerKey,
    apply_writes,
    finish_channels,
    plan_tasks,
)
from pregel_redux.shutdown import RunAborted, ShutdownController, ShutdownSignal, read_shutdown
from pregel_redux.snapshots import PullPath, PushPath, StateSnapshot, TaskSnapshot
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
)

type AnyGraph = CompiledGraph[object, Never, object]


def dump_trigger_versions(versions: Mapping[StartKey | TriggerKey, int]) -> TriggerVersions:
    return TriggerVersions(
        branches={key.node: version for key, version in versions.items() if isinstance(key, BranchKey)},
        joins=tuple(
            JoinVersion(sources=tuple(sorted(key.sources)), target=key.target, version=version)
            for key, version in versions.items()
            if isinstance(key, JoinKey)
        ),
    )


def load_trigger_versions(versions: TriggerVersions) -> dict[StartKey | TriggerKey, int]:
    return {
        **{BranchKey(node): version for node, version in versions.branches.items()},
        **{JoinKey(frozenset(join.sources), join.target): join.version for join in versions.joins},
    }


def dump_send(graph: AnyGraph, send: Send) -> PendingSend:
    node = graph.nodes[send.node]
    return PendingSend(node=send.node, node_input=dump_json(node.input_adapter, node.build_send_input(send)))


def load_send(graph: AnyGraph, send: PendingSend) -> Send:
    return Send(send.node, load_json(graph.nodes[send.node].input_adapter, send.node_input))


def dump_updates(graph: AnyGraph, updates: Mapping[str, object]) -> dict[str, JsonValue]:
    return {name: dump_json(graph.channels[name].update_adapter, value) for name, value in updates.items()}


def load_updates(graph: AnyGraph, updates: Mapping[str, JsonValue]) -> dict[str, object]:
    reject_unknown_channels(graph.channels, updates)
    return {name: load_json(graph.channels[name].update_adapter, value) for name, value in updates.items()}


def dump_writes(graph: AnyGraph, writes: TaskWrites) -> SavedWrites:
    return SavedWrites(
        fields=dump_updates(graph, writes.fields),
        branches=tuple(sorted(writes.branches)),
        arrivals=tuple(
            SavedArrival(sources=tuple(sorted(arrival.join.sources)), target=arrival.join.target, source=arrival.source)
            for arrival in writes.arrivals
        ),
        sends=tuple(dump_send(graph, send) for send in writes.sends),
    )


def load_writes(graph: AnyGraph, saved: SavedWrites) -> TaskWrites:
    return TaskWrites(
        fields=load_updates(graph, saved.fields),
        branches=frozenset(saved.branches),
        arrivals=tuple(
            Arrival(JoinKey(frozenset(arrival.sources), arrival.target), arrival.source) for arrival in saved.arrivals
        ),
        sends=tuple(load_send(graph, send) for send in saved.sends),
    )


def dump_branch(node: str, channel: BranchChannel) -> SavedBranch | None:
    match channel:
        case EphemeralValue(stored=Stored()):
            return SavedBranch(node=node, finished=None)
        case LastValueAfterFinish(stored=Stored()):
            return SavedBranch(node=node, finished=channel.finished)
        case _:
            return None


def load_branch(channel: BranchChannel, saved: SavedBranch) -> BranchChannel:
    match channel:
        case EphemeralValue():
            return EphemeralValue(Stored(None))
        case LastValueAfterFinish():
            return LastValueAfterFinish(Stored(None), finished=bool(saved.finished))
        case _:
            assert_never(channel)


def dump_join(key: JoinKey, channel: JoinChannel) -> SavedJoin:
    match channel:
        case NamedBarrierValue():
            finished = None
        case NamedBarrierValueAfterFinish():
            finished = channel.finished
        case _:
            assert_never(channel)
    return SavedJoin(
        sources=tuple(sorted(key.sources)), target=key.target, seen=tuple(sorted(channel.seen)), finished=finished
    )


def load_join(channel: JoinChannel, saved: SavedJoin) -> JoinChannel:
    match channel:
        case NamedBarrierValue():
            return NamedBarrierValue(channel.names, frozenset(saved.seen))
        case NamedBarrierValueAfterFinish():
            return NamedBarrierValueAfterFinish(channel.names, frozenset(saved.seen), finished=bool(saved.finished))
        case _:
            assert_never(channel)


@dataclass(frozen=True)
class CheckpointStamp:
    checkpoint_id: UUID
    parent_id: UUID | None
    created_at: datetime


def make_checkpoint_id(parent_id: UUID | None) -> UUID:
    milliseconds = time.time_ns() // 1_000_000
    layout = (milliseconds << 80) | (7 << 76) | (secrets.randbits(12) << 64) | (2 << 62) | secrets.randbits(62)
    candidate = UUID(int=layout)
    return candidate if parent_id is None or candidate > parent_id else UUID(int=parent_id.int + 1)


def dump_position(
    graph: AnyGraph, position: Position, stamp: CheckpointStamp, checkpoint_ns: str, step: int, source: CheckpointSource
) -> Checkpoint:
    channels = position.channels
    versions = position.versions
    branches = (dump_branch(node, channel) for node, channel in channels.branches.items())
    return Checkpoint(
        id=stamp.checkpoint_id,
        parent_id=stamp.parent_id,
        ts=stamp.created_at,
        checkpoint_ns=checkpoint_ns,
        step=step,
        source=source,
        channel_values=ChannelValues(
            start=None if channels.start.stored is None else dump_updates(graph, channels.start.stored.value),
            fields={name: saved.value for name, channel in channels.fields.items() if (saved := channel.checkpoint())},
            branches=tuple(branch for branch in branches if branch is not None),
            joins=tuple(dump_join(key, channel) for key, channel in channels.joins.items() if channel.seen),
            tasks=tuple(dump_send(graph, send) for send in channels.tasks.values),
        ),
        channel_versions=ChannelVersions(
            start=versions.get(StartKey()),
            fields={key.name: version for key, version in versions.items() if isinstance(key, FieldKey)},
            triggers=dump_trigger_versions(
                {key: version for key, version in versions.items() if isinstance(key, BranchKey | JoinKey)}
            ),
            tasks=versions.get(TasksKey()),
        ),
        versions_seen={
            node: dump_trigger_versions(seen) for node, seen in position.seen.items() if isinstance(node, str)
        },
        start_seen=position.seen.get(Endpoint.start, {}).get(StartKey()),
    )


def load_position(graph: AnyGraph, checkpoint: Checkpoint) -> Position:
    values = checkpoint.channel_values
    reject_unknown_channels(graph.channels, values.fields)
    opened = graph.open({}).channels
    joins = {JoinKey(frozenset(join.sources), join.target): join for join in values.joins}
    unknown_nodes = sorted(
        {*(branch.node for branch in values.branches), *checkpoint.versions_seen, *(send.node for send in values.tasks)}
        - graph.nodes.keys()
    )
    unknown_joins = [key for key in joins if key not in opened.joins]
    if unknown_nodes or unknown_joins:
        raise ValueError(
            f"the checkpoint refers to nodes {unknown_nodes!r} and joins {unknown_joins!r}, which are not in this graph"
        )
    channels = Channels(
        start=EphemeralValue(None if values.start is None else Stored(load_updates(graph, values.start))),
        fields={
            **opened.fields,
            **{name: opened.fields[name].from_checkpoint(saved) for name, saved in values.fields.items()},
        },
        branches={
            **opened.branches,
            **{branch.node: load_branch(opened.branches[branch.node], branch) for branch in values.branches},
        },
        joins={**opened.joins, **{key: load_join(opened.joins[key], join) for key, join in joins.items()}},
        tasks=Topic(tuple(load_send(graph, send) for send in values.tasks)),
    )
    versions: dict[ChannelKey, int] = {
        **({} if checkpoint.channel_versions.start is None else {StartKey(): checkpoint.channel_versions.start}),
        **{FieldKey(name): version for name, version in checkpoint.channel_versions.fields.items()},
        **load_trigger_versions(checkpoint.channel_versions.triggers),
        **({} if checkpoint.channel_versions.tasks is None else {TasksKey(): checkpoint.channel_versions.tasks}),
    }
    seen: dict[NodeName, Mapping[StartKey | TriggerKey, int]] = {
        **{node: load_trigger_versions(seen) for node, seen in checkpoint.versions_seen.items()},
        **({} if checkpoint.start_seen is None else {Endpoint.start: {StartKey(): checkpoint.start_seen}}),
    }
    return Position(channels, versions, seen)


def read_command_writes(graph: AnyGraph, command: Command[str]) -> TaskWrites:
    routes = resolve_goto(command.goto, graph.nodes.keys())
    return TaskWrites(parse_updates(graph.channels, extract_writes(command.update)), routes.targets, (), routes.sends)


def apply_command(graph: AnyGraph, position: Position, writes: TaskWrites) -> Position:
    sends = (*position.channels.tasks.values, *writes.sends) if writes.sends else ()
    branches = frozenset(node for node in writes.branches if not position.channels.branches[node].is_available())
    applied = apply_writes(position, [], [replace(writes, branches=branches, sends=sends)], [], graph.trigger_keys)
    return applied if plan_tasks(applied, graph.triggers) else finish_channels(applied)


@dataclass(frozen=True)
class ThreadScope:
    graph: AnyGraph
    checkpointer: AsyncCheckpointer
    thread_id: str
    checkpoint_ns: str

    def read_position(self, saved: CheckpointTuple) -> Position:
        position = load_position(self.graph, saved.checkpoint)
        commands = sorted(
            (write for write in saved.pending_writes if isinstance(write, RunCommandRecord)),
            key=lambda record: record.index,
        )
        for record in commands:
            position = apply_command(self.graph, position, load_writes(self.graph, record.command))
        return position

    async def put(
        self, position: Position, step: int, source: CheckpointSource, parent_id: UUID | None, newest_id: UUID | None
    ) -> Checkpoint:
        stamp = CheckpointStamp(make_checkpoint_id(newest_id), parent_id, datetime.now(UTC))
        checkpoint = dump_position(self.graph, position, stamp, self.checkpoint_ns, step, source)
        await finish_despite_cancellation(self.checkpointer.aput(self.thread_id, checkpoint))
        return checkpoint

    async def begin(self, position: Position, step: int, parent_id: UUID | None) -> "ThreadJournal":
        checkpoint = await self.put(position, step, CheckpointSource.input, parent_id, parent_id)
        return ThreadJournal(self, CheckpointTuple(checkpoint=checkpoint, pending_writes=()), checkpoint.id)

    async def find(self, checkpoint_id: UUID | None) -> "Loaded | None":
        newest = await self.checkpointer.aget_tuple(self.thread_id, self.checkpoint_ns, None)
        if checkpoint_id is None:
            return None if newest is None else Loaded(newest, newest.checkpoint.id)
        saved = await self.checkpointer.aget_tuple(self.thread_id, self.checkpoint_ns, checkpoint_id)
        if saved is None or newest is None:
            raise LookupError(f"thread {self.thread_id!r} has no checkpoint {checkpoint_id}")
        return Loaded(saved, newest.checkpoint.id)

    async def open_run(self, checkpoint_id: UUID | None, *, answers: bool) -> "OpenedRun | None":
        loaded = await self.find(checkpoint_id)
        if loaded is None:
            return None
        saved = loaded.saved
        position = self.read_position(saved)
        if checkpoint_id is None:
            return OpenedRun(position, ThreadJournal(self, saved, loaded.newest_id))
        if answers:
            superseded = saved.checkpoint.id != loaded.newest_id
            kept = tuple(
                write for write in saved.pending_writes if not (superseded and isinstance(write, TaskWritesRecord))
            )
            replayed = CheckpointTuple(checkpoint=saved.checkpoint, pending_writes=kept)
            return OpenedRun(position, ThreadJournal(self, replayed, loaded.newest_id))
        step = saved.checkpoint.step
        fork = await self.put(position, step, CheckpointSource.fork, checkpoint_id, loaded.newest_id)
        return OpenedRun(position, ThreadJournal(self, CheckpointTuple(checkpoint=fork, pending_writes=()), fork.id))

    async def read_child_snapshot(self, journal: "ThreadJournal", task: TaskRef) -> "StateSnapshot | None":
        child = self.graph.nodes[task.node].action if isinstance(task.node, str) else None
        if not isinstance(child, CompiledGraph):
            return None
        scope = ThreadScope(child, self.checkpointer, self.thread_id, journal.make_task_ns(task))
        saved = await self.checkpointer.aget_tuple(scope.thread_id, scope.checkpoint_ns, None)
        return None if saved is None else await scope.read_snapshot(saved, subgraphs=True)

    async def read_task_snapshot(self, journal: "ThreadJournal", task: TaskRef, *, subgraphs: bool) -> TaskSnapshot:
        task_id = journal.make_task_id(task)
        name = str(task.node) if isinstance(task.node, str) else "__start__"
        writes = journal.find_writes(task)
        child = await self.read_child_snapshot(journal, task) if subgraphs or writes is None else None
        stop = journal.find_stop(task)
        if writes is not None:
            waiting: tuple[Interrupt, ...] = ()
        elif child is not None:
            waiting = child.interrupts
        else:
            waiting = journal.find_waiting(task)
        match stop:
            case TaskFailed(error=failure) if writes is None:
                error = failure
            case _:
                error = None
        return TaskSnapshot(
            id=task_id,
            name=name,
            path=PullPath(name) if task.send_index is None else PushPath(task.send_index),
            interrupts=waiting,
            error=error,
            result=None if writes is None else writes.fields,
            state=child if subgraphs else None,
        )

    async def read_snapshot(self, saved: CheckpointTuple, *, subgraphs: bool) -> "StateSnapshot":
        position = self.read_position(saved)
        journal = ThreadJournal(self, saved, saved.checkpoint.id)
        refs = [refer_task(task) for task in plan_tasks(position, self.graph.triggers)]
        tasks = tuple([await self.read_task_snapshot(journal, ref, subgraphs=subgraphs) for ref in refs])
        return StateSnapshot(
            values=position.channels.read_values(),
            next=tuple(task.name for task in tasks),
            checkpoint_ns=self.checkpoint_ns,
            checkpoint_id=saved.checkpoint.id,
            parent_id=saved.checkpoint.parent_id,
            created_at=saved.checkpoint.ts,
            step=saved.checkpoint.step,
            source=saved.checkpoint.source,
            tasks=tasks,
            interrupts=tuple(item for task in tasks for item in task.interrupts),
        )


@dataclass(frozen=True)
class Loaded:
    saved: CheckpointTuple
    newest_id: UUID


@dataclass
class ThreadJournal:
    scope: ThreadScope
    saved: CheckpointTuple
    newest_id: UUID

    @property
    def ns(self) -> tuple[str, ...]:
        return tuple(self.scope.checkpoint_ns.split("|")) if self.scope.checkpoint_ns else ()

    def make_task_id(self, task: TaskRef) -> str:
        checkpoint = self.saved.checkpoint
        return str(uuid5(checkpoint.id, f"{self.scope.checkpoint_ns}|{checkpoint.step + 1}|{name_task(task)}"))

    async def save(self, position: Position, source: CheckpointSource) -> None:
        parent = self.saved.checkpoint
        checkpoint = await self.scope.put(position, parent.step + 1, source, parent.id, self.newest_id)
        self.saved = CheckpointTuple(checkpoint=checkpoint, pending_writes=())
        self.newest_id = checkpoint.id

    async def read_snapshot(self) -> StateSnapshot:
        return await self.scope.read_snapshot(self.saved, subgraphs=False)

    def make_task_ns(self, task: TaskRef) -> str:
        parent_ns = f"{self.scope.checkpoint_ns}|" if self.scope.checkpoint_ns else ""
        return f"{parent_ns}{task.node}:{self.make_task_id(task)}"

    async def put_write(self, write: PendingWrite) -> None:
        scope = self.scope
        checkpoint = self.saved.checkpoint
        await finish_despite_cancellation(
            scope.checkpointer.aput_writes(scope.thread_id, scope.checkpoint_ns, checkpoint.id, write)
        )
        kept = (kept for kept in self.saved.pending_writes if kept.key != write.key)
        self.saved = CheckpointTuple(checkpoint=checkpoint, pending_writes=(*kept, write))

    async def record(self, task: TaskRef, writes: TaskWrites) -> None:
        saved = dump_writes(self.scope.graph, writes)
        await self.put_write(TaskWritesRecord(task_id=self.make_task_id(task), writes=saved))

    async def record_waiting(self, task: TaskRef, interrupts: Sequence[Interrupt]) -> None:
        stop = TaskWaiting(interrupts=tuple(interrupts))
        await self.put_write(TaskStopRecord(task_id=self.make_task_id(task), stop=stop))

    async def record_failure(self, task: TaskRef, error: str) -> None:
        earlier = self.find_stop(task)
        stop = TaskFailed(error=error, interrupts=() if earlier is None else earlier.interrupts)
        await self.put_write(TaskStopRecord(task_id=self.make_task_id(task), stop=stop))

    async def record_answers(self, task: TaskRef, answers: Sequence[JsonValue]) -> None:
        await self.put_write(TaskAnswersRecord(task_id=self.make_task_id(task), answers=tuple(answers)))

    def find_writes(self, task: TaskRef) -> TaskWrites | None:
        task_id = self.make_task_id(task)
        saved = [
            write.writes
            for write in self.saved.pending_writes
            if isinstance(write, TaskWritesRecord) and write.task_id == task_id
        ]
        return load_writes(self.scope.graph, saved[0]) if saved else None

    def find_stop(self, task: TaskRef) -> TaskWaiting | TaskFailed | None:
        task_id = self.make_task_id(task)
        stops = [
            write.stop
            for write in self.saved.pending_writes
            if isinstance(write, TaskStopRecord) and write.task_id == task_id
        ]
        return stops[0] if stops else None

    def find_answers(self, task: TaskRef) -> tuple[JsonValue, ...]:
        task_id = self.make_task_id(task)
        return tuple(
            answer
            for write in self.saved.pending_writes
            if isinstance(write, TaskAnswersRecord) and write.task_id == task_id
            for answer in write.answers
        )

    def find_waiting(self, task: TaskRef) -> tuple[Interrupt, ...]:
        stop = self.find_stop(task)
        if stop is None:
            return ()
        task_ns = self.make_task_ns(task)
        answered = {make_interrupt_id(task_ns, index) for index in range(len(self.find_answers(task)))}
        return tuple(interrupt for interrupt in stop.interrupts if interrupt.id not in answered)

    def open_scratchpad(self, task: TaskRef, resume: ResumeInput) -> Scratchpad:
        return Scratchpad(self.make_task_ns(task), list(self.find_answers(task)), resume, 0)

    async def read_resume(self, command: Command[str] | None) -> ResumeInput:
        if command is None:
            return ResumeInput([], {})
        given = command.resume_by_interrupt_id or {}
        if command.resume is not None:
            waiting = [interrupt.id for interrupt in (await self.read_snapshot()).interrupts]
            match waiting:
                case []:
                    return ResumeInput([command.resume], given)
                case [interrupt_id]:
                    return ResumeInput([], {**given, interrupt_id: command.resume})
                case _:
                    raise ValueError(f"the interrupts {waiting!r} wait for an answer, so give resume_by_interrupt_id")
        return ResumeInput([], given)

    async def record_command(self, writes: TaskWrites) -> None:
        index = sum(1 for write in self.saved.pending_writes if isinstance(write, RunCommandRecord))
        await self.put_write(RunCommandRecord(index=index, command=dump_writes(self.scope.graph, writes)))

    async def enter_child(self, child: AnyGraph, call: NodeCall) -> ChildEntry:
        scope = ThreadScope(child, self.scope.checkpointer, self.scope.thread_id, self.make_task_ns(call.task))
        opened = await scope.open_run(None, answers=False)
        if opened is not None:
            return ChildEntry(opened.position, opened.journal, resumed=True)
        start = child.feed(child.open(extract_writes(call.node_input)), {})
        return ChildEntry(start, await scope.begin(start, -1, None), resumed=False)


@dataclass(frozen=True)
class OpenedRun:
    position: Position
    journal: ThreadJournal


def infer_writer(graph: AnyGraph, versions_seen: Mapping[str, TriggerVersions]) -> str:
    if len(graph.nodes) == 1:
        return next(iter(graph.nodes))
    last_seen = sorted(
        (max(versions), node)
        for node, seen in versions_seen.items()
        if node in graph.nodes and (versions := [*seen.branches.values(), *(join.version for join in seen.joins)])
    )
    if len(last_seen) == 1 or (len(last_seen) > 1 and last_seen[-1][0] != last_seen[-2][0]):
        return last_seen[-1][1]
    raise ValueError("the node that made the last update is not clear, so give as_node")


def find_graph(graph: AnyGraph, checkpoint_ns: str) -> AnyGraph:
    found = graph
    for part in checkpoint_ns.split("|") if checkpoint_ns else ():
        name = part.rsplit(":", 1)[0]
        match found.nodes[name].action if name in found.nodes else None:
            case CompiledGraph() as child:
                found = child
            case _:
                raise LookupError(f"the namespace {checkpoint_ns!r} names {name!r}, which is not a child graph")
    return found


@dataclass(frozen=True)
class RunStart:
    position: Position
    journal: ThreadJournal
    resume: ResumeInput
    resumed: bool


@dataclass(frozen=True)
class PersistentGraph[StateT, InputT, OutputT]:
    graph: CompiledGraph[StateT, InputT, OutputT]
    checkpointer: Checkpointer

    async def start(
        self,
        checkpointer: AsyncCheckpointer,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
    ) -> RunStart:
        target = parse_run_target(config)
        thread_id = target.thread_id
        scope = ThreadScope(self.graph, checkpointer, thread_id, "")
        match input:
            case Command() as command:
                answers = command.resume is not None or bool(command.resume_by_interrupt_id)
            case _:
                answers = False
        opened = await scope.open_run(target.fork_from, answers=answers)
        match input:
            case None | Command():
                if opened is None:
                    raise LookupError(f"thread {thread_id!r} has no checkpoint, so the run needs an input")
                position, journal = opened.position, opened.journal
                resume = await journal.read_resume(input)
                writes = None if input is None else read_command_writes(self.graph, input)
                if writes is not None and (writes.fields or writes.branches or writes.sends):
                    await journal.record_command(writes)
                    position = apply_command(self.graph, position, writes)
                return RunStart(position, journal, resume, resumed=True)
            case _ if opened is None:
                position = self.graph.feed(self.graph.open({}), extract_writes(input))
                journal = await scope.begin(position, -1, None)
            case _:
                position = self.graph.feed(opened.position, extract_writes(input))
                checkpoint = opened.journal.saved.checkpoint
                journal = await scope.begin(position, checkpoint.step + 1, checkpoint.id)
        return RunStart(position, journal, ResumeInput([], {}), resumed=False)

    async def run(
        self,
        checkpointer: AsyncCheckpointer,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        step_limit: int | None,
        stream: Stream,
        signal: ShutdownSignal | None,
    ) -> Finished[OutputT] | Interrupted:
        shutdown = ShutdownController().signal if signal is None else signal
        if shutdown.aborted:
            raise RunAborted("the shutdown signal of the run is aborted")
        return await self.run_thread(checkpointer, input, config, step_limit, stream, shutdown)

    async def run_thread(
        self,
        checkpointer: AsyncCheckpointer,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        step_limit: int | None,
        stream: Stream,
        shutdown: ShutdownSignal,
    ) -> Finished[OutputT] | Interrupted:
        start = await self.start(checkpointer, input, config)
        provided: dict[Depends, object] = {
            **{Depends(read_config, scope=scope): config for scope in Scope},
            **{Depends(read_shutdown, scope=scope): shutdown for scope in Scope},
        }
        return await self.graph.run(
            start.position, step_limit, start.journal, start.resume, stream, provided, shutdown, resumed=start.resumed
        )

    async def ainvoke(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Finished[OutputT] | Interrupted:
        stream = Stream(frozenset(), subgraphs=False, emit=lambda part: None)
        return await self.run(self.checkpointer, input, config, step_limit, stream, signal)

    def invoke(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Finished[OutputT] | Interrupted:
        self.graph.reject_async_nodes("ainvoke")
        stream = Stream(frozenset(), subgraphs=False, emit=lambda part: None)
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return run_in_new_loop(lambda: self.run(checkpointer, input, config, step_limit, stream, signal))

    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.values],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[ValuesPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.updates],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[UpdatesPart | InterruptsPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.custom],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[CustomPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.tasks],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[TaskStartPart | TaskResultPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.checkpoints],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[CheckpointPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[StreamPart]: ...
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: StreamMode | Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[StreamPart]:
        modes = frozenset([stream_mode] if isinstance(stream_mode, StreamMode) else stream_mode)
        return stream_run(
            lambda emit: self.run(
                self.checkpointer,
                input,
                config,
                step_limit,
                Stream(modes, subgraphs=subgraphs, emit=emit),
                signal,
            )
        )

    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.values],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[ValuesPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.updates],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[UpdatesPart | InterruptsPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.custom],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[CustomPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.tasks],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[TaskStartPart | TaskResultPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.checkpoints],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[CheckpointPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[StreamPart]: ...
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: StreamMode | Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[StreamPart]:
        self.graph.reject_async_nodes("astream")
        modes = frozenset([stream_mode] if isinstance(stream_mode, StreamMode) else stream_mode)
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return iterate_in_new_loop(
            lambda: stream_run(
                lambda emit: self.run(
                    checkpointer,
                    input,
                    config,
                    step_limit,
                    Stream(modes, subgraphs=subgraphs, emit=emit),
                    signal,
                )
            )
        )

    async def read_state(
        self, checkpointer: AsyncCheckpointer, config: ThreadConfig, *, subgraphs: bool
    ) -> StateSnapshot:
        address = parse_state_address(config)
        graph = find_graph(self.graph, address.checkpoint_ns)
        scope = ThreadScope(graph, checkpointer, address.thread_id, address.checkpoint_ns)
        saved = await checkpointer.aget_tuple(address.thread_id, address.checkpoint_ns, address.checkpoint_id)
        if saved is None:
            raise LookupError(f"thread {address.thread_id!r} has no such checkpoint")
        return await scope.read_snapshot(saved, subgraphs=subgraphs)

    async def aget_state(self, config: ThreadConfig, *, subgraphs: bool = False) -> StateSnapshot:
        return await self.read_state(self.checkpointer, config, subgraphs=subgraphs)

    def get_state(self, config: ThreadConfig, *, subgraphs: bool = False) -> StateSnapshot:
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return run_in_new_loop(lambda: self.read_state(checkpointer, config, subgraphs=subgraphs))

    async def read_state_history(
        self, checkpointer: AsyncCheckpointer, config: ThreadConfig, before: UUID | None, limit: int | None
    ) -> AsyncIterator[StateSnapshot]:
        address = parse_state_address(config)
        thread_id, checkpoint_ns = address.thread_id, address.checkpoint_ns
        if limit is not None and limit < 0:
            raise ValueError(f"limit is {limit}, and the number of checkpoints to read cannot be negative")
        scope = ThreadScope(find_graph(self.graph, checkpoint_ns), checkpointer, thread_id, checkpoint_ns)
        if address.checkpoint_id is not None:
            saved = await checkpointer.aget_tuple(thread_id, checkpoint_ns, address.checkpoint_id)
            if saved is not None and (before is None or saved.checkpoint.id < before) and limit != 0:
                yield await scope.read_snapshot(saved, subgraphs=False)
            return
        async for saved in checkpointer.alist(thread_id, checkpoint_ns, before=before, limit=limit):
            yield await scope.read_snapshot(saved, subgraphs=False)

    def aget_state_history(
        self, config: ThreadConfig, *, before: UUID | None = None, limit: int | None = None
    ) -> AsyncIterator[StateSnapshot]:
        return self.read_state_history(self.checkpointer, config, before, limit)

    def get_state_history(
        self, config: ThreadConfig, *, before: UUID | None = None, limit: int | None = None
    ) -> Iterator[StateSnapshot]:
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return iterate_in_new_loop(lambda: self.read_state_history(checkpointer, config, before, limit))

    async def write_update(
        self, checkpointer: AsyncCheckpointer, config: ThreadConfig, values: object, as_node: str | None
    ) -> UUID:
        address = parse_state_address(config)
        graph = find_graph(self.graph, address.checkpoint_ns)
        scope = ThreadScope(graph, checkpointer, address.thread_id, address.checkpoint_ns)
        loaded = await scope.find(address.checkpoint_id)
        finished: list[tuple[Task, TaskWrites]] = []
        if loaded is None:
            node = infer_writer(graph, {}) if as_node is None else as_node
            position, step, parent_id, newest_id = graph.open({}), -1, None, None
        else:
            checkpoint = loaded.saved.checkpoint
            node = infer_writer(graph, checkpoint.versions_seen) if as_node is None else as_node
            position = scope.read_position(loaded.saved)
            if address.checkpoint_id is None:
                journal = ThreadJournal(scope, loaded.saved, loaded.newest_id)
                finished = [
                    (task, writes)
                    for task in plan_tasks(position, graph.triggers)
                    if (writes := journal.find_writes(refer_task(task))) is not None
                ]
            step, parent_id, newest_id = checkpoint.step, checkpoint.id, loaded.newest_id
        if node not in graph.nodes:
            raise ValueError(f"as_node is {node!r}, which is not a node of this graph")
        updated = apply_writes(
            position,
            [*(task for task, _ in finished), PullTask(node, ())],
            [*(writes for _, writes in finished), graph.read_writes(node, values)],
            [],
            graph.trigger_keys,
        )
        saved = await scope.put(updated, step + 1, CheckpointSource.update, parent_id, newest_id)
        return saved.id

    async def aupdate_state(self, config: ThreadConfig, values: object, *, as_node: str | None = None) -> UUID:
        return await self.write_update(self.checkpointer, config, values, as_node)

    def update_state(self, config: ThreadConfig, values: object, *, as_node: str | None = None) -> UUID:
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return run_in_new_loop(lambda: self.write_update(checkpointer, config, values, as_node))

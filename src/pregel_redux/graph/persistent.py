import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Never, assert_never
from uuid import UUID, uuid5

from pydantic import JsonValue

from pregel_redux.channels import Stored, dump_json, extract_writes, load_json, reject_unknown_channels
from pregel_redux.checkpoints import (
    ChannelValues,
    ChannelVersions,
    Checkpoint,
    Checkpointer,
    CheckpointSource,
    JoinVersion,
    PendingSend,
    PendingWrite,
    RunResumeRecord,
    SavedArrival,
    SavedBranch,
    SavedJoin,
    SavedWrites,
    TaskInterruptRecord,
    TaskWritesRecord,
    TriggerVersions,
)
from pregel_redux.commands import Command, Endpoint, Send
from pregel_redux.graph.compiled import CompiledGraph, Journal, NodeCall, TaskRef
from pregel_redux.interrupts import Finished, Interrupt, Interrupted, ResumeInput, Scratchpad
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
    StartKey,
    TasksKey,
    TaskWrites,
    Topic,
    TriggerKey,
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


def name_task(task: TaskRef) -> str:
    match task.node:
        case Endpoint.start:
            return "start"
        case str() as node:
            return f"pull|{node}" if task.send_index is None else f"push|{node}|{task.send_index}"
        case _:
            assert_never(task.node)


@dataclass(frozen=True)
class ThreadScope:
    graph: AnyGraph
    checkpointer: Checkpointer
    thread_id: str
    checkpoint_ns: str

    async def put(self, position: Position, step: int, source: CheckpointSource, parent_id: UUID | None) -> UUID:
        stamp = CheckpointStamp(make_checkpoint_id(parent_id), parent_id, datetime.now(UTC))
        checkpoint = dump_position(self.graph, position, stamp, self.checkpoint_ns, step, source)
        await self.checkpointer.aput(self.thread_id, checkpoint)
        return stamp.checkpoint_id

    async def begin(self, position: Position, step: int, parent_id: UUID | None) -> "tuple[Position, ThreadJournal]":
        checkpoint_id = await self.put(position, step, CheckpointSource.input, parent_id)
        return position, ThreadJournal(self, checkpoint_id, step + 1, ())

    async def load(self) -> "tuple[Position, ThreadJournal] | None":
        saved = await self.checkpointer.aget_tuple(self.thread_id, self.checkpoint_ns)
        if saved is None:
            return None
        journal = ThreadJournal(self, saved.checkpoint.id, saved.checkpoint.step + 1, saved.pending_writes)
        return load_position(self.graph, saved.checkpoint), journal


@dataclass
class ThreadJournal:
    scope: ThreadScope
    checkpoint_id: UUID
    step: int
    pending: Sequence[PendingWrite]

    def make_task_id(self, task: TaskRef) -> str:
        return str(uuid5(self.checkpoint_id, f"{self.scope.checkpoint_ns}|{self.step}|{name_task(task)}"))

    async def save(self, position: Position, source: CheckpointSource) -> None:
        self.checkpoint_id = await self.scope.put(position, self.step, source, self.checkpoint_id)
        self.step += 1
        self.pending = ()

    def make_task_ns(self, task: TaskRef) -> str:
        parent_ns = f"{self.scope.checkpoint_ns}|" if self.scope.checkpoint_ns else ""
        return f"{parent_ns}{task.node}:{self.make_task_id(task)}"

    async def put_write(self, write: PendingWrite) -> None:
        scope = self.scope
        await scope.checkpointer.aput_writes(scope.thread_id, scope.checkpoint_ns, self.checkpoint_id, write)

    async def record(self, task: TaskRef, writes: TaskWrites) -> None:
        saved = dump_writes(self.scope.graph, writes)
        await self.put_write(TaskWritesRecord(task_id=self.make_task_id(task), writes=saved))

    async def record_interrupt(
        self, task: TaskRef, interrupts: Sequence[Interrupt], resume: Sequence[JsonValue]
    ) -> None:
        await self.put_write(
            TaskInterruptRecord(task_id=self.make_task_id(task), interrupts=tuple(interrupts), resume=tuple(resume))
        )

    def find_writes(self, task: TaskRef) -> TaskWrites | None:
        task_id = self.make_task_id(task)
        saved = [
            write.writes for write in self.pending if isinstance(write, TaskWritesRecord) and write.task_id == task_id
        ]
        return load_writes(self.scope.graph, saved[0]) if saved else None

    def open_scratchpad(self, task: TaskRef, resume: ResumeInput) -> Scratchpad:
        task_id = self.make_task_id(task)
        saved = [
            value
            for write in self.pending
            if isinstance(write, TaskInterruptRecord) and write.task_id == task_id
            for value in write.resume
        ]
        return Scratchpad(self.make_task_ns(task), saved, resume, 0)

    async def read_resume(self, command: Command[str] | None) -> ResumeInput:
        if command is not None and command.resume is not None:
            finished = {write.task_id for write in self.pending if isinstance(write, TaskWritesRecord)}
            waiting = [
                interrupt.id
                for write in self.pending
                if isinstance(write, TaskInterruptRecord) and write.task_id not in finished
                for interrupt in write.interrupts
            ]
            if len(waiting) > 1:
                raise ValueError(f"the interrupts {waiting!r} wait for an answer, so give resume_by_interrupt_id")
            await self.put_write(RunResumeRecord(value=command.resume))
            return ResumeInput([command.resume], command.resume_by_interrupt_id or {})
        saved = [write.value for write in self.pending if isinstance(write, RunResumeRecord)]
        return ResumeInput(saved, {} if command is None else command.resume_by_interrupt_id or {})

    async def enter_child(self, child: AnyGraph, call: NodeCall) -> tuple[Position, Journal]:
        scope = ThreadScope(child, self.scope.checkpointer, self.scope.thread_id, self.make_task_ns(call.task))
        return await scope.load() or await scope.begin(
            child.feed(child.open(extract_writes(call.node_input)), {}), -1, None
        )


@dataclass(frozen=True)
class PersistentGraph[StateT, InputT, OutputT]:
    graph: CompiledGraph[StateT, InputT, OutputT]
    checkpointer: Checkpointer

    async def ainvoke(
        self, input: InputT | Command[str] | None, *, thread_id: str, step_limit: int | None = None
    ) -> Finished[OutputT] | Interrupted:
        scope = ThreadScope(self.graph, self.checkpointer, thread_id, "")
        loaded = await scope.load()
        match input:
            case None | Command():
                if loaded is None:
                    raise LookupError(f"thread {thread_id!r} has no checkpoint, so the run needs an input")
                position, journal = loaded
                resume = await journal.read_resume(input)
            case _ if loaded is None:
                start = self.graph.feed(self.graph.open({}), extract_writes(input))
                position, journal = await scope.begin(start, -1, None)
                resume = ResumeInput([], {})
            case _:
                start = self.graph.feed(loaded[0], extract_writes(input))
                position, journal = await scope.begin(start, loaded[1].step, loaded[1].checkpoint_id)
                resume = ResumeInput([], {})
        return await self.graph.run(position, step_limit, journal, resume)

    def invoke(
        self, input: InputT | Command[str] | None, *, thread_id: str, step_limit: int | None = None
    ) -> Finished[OutputT] | Interrupted:
        return self.graph.run_in_new_loop(
            lambda: self.ainvoke(input, thread_id=thread_id, step_limit=step_limit), "ainvoke"
        )

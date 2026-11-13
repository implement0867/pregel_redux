from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid5

from pydantic import JsonValue

from pregel_redux.channels import (
    extract_writes,
)
from pregel_redux.checkpoints import (
    AsyncCheckpointer,
    Checkpoint,
    CheckpointSource,
    CheckpointTuple,
    PendingWrite,
    RunCommandRecord,
    TaskAnswersRecord,
    TaskFailed,
    TaskStopRecord,
    TaskWaiting,
    TaskWritesRecord,
    TriggerVersions,
)
from pregel_redux.commands import Command
from pregel_redux.dependencies import finish_despite_cancellation
from pregel_redux.graph.compiled import ChildEntry, CompiledGraph
from pregel_redux.graph.conversion import (
    AnyGraph,
    CheckpointStamp,
    dump_position,
    dump_writes,
    load_position,
    load_writes,
    make_checkpoint_id,
)
from pregel_redux.graph.task_refs import NodeCall, TaskRef, name_task, refer_task
from pregel_redux.interrupts import Interrupt, ResumeInput, Scratchpad, make_interrupt_id
from pregel_redux.pregel import (
    Position,
    TaskWrites,
    apply_command,
    plan_tasks,
)
from pregel_redux.snapshots import PullPath, PushPath, StateSnapshot, TaskSnapshot


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
            position = apply_command(
                self.graph.trigger_keys, self.graph.triggers, position, load_writes(self.graph, record.command)
            )
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

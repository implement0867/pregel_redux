from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import assert_never
from uuid import UUID, uuid5

from typing_extensions import TypeForm

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
    TaskFailed,
    TaskStopRecord,
    TaskWaiting,
    TaskWritesRecord,
    TriggerVersions,
)
from pregel_redux.configs import Namespace, Segment, encode_namespace
from pregel_redux.dependencies import CallPlan, finish_despite_cancellation
from pregel_redux.graph.compiled import CompiledGraph, Continuation, FreshStart, Start
from pregel_redux.graph.conversion import (
    AnyGraph,
    CheckpointStamp,
    dump_position,
    dump_writes,
    load_position,
    load_writes,
    make_checkpoint_id,
)
from pregel_redux.messages import Entrance, Message
from pregel_redux.pregel import (
    Position,
    TaskWrites,
    apply_command,
    deliver,
    plan_tasks,
)
from pregel_redux.snapshots import (
    FailedTask,
    FinishedTask,
    PendingTask,
    StateSnapshot,
    TaskSnapshot,
    TaskStatus,
    WaitingTask,
)
from pregel_redux.streams import CheckpointPart
from pregel_redux.task_refs import NodeCall, PullRef, PushRef, StartRef, TaskRef, name_task, refer_task


@dataclass(frozen=True)
class ThreadScope:
    graph: AnyGraph
    checkpointer: AsyncCheckpointer
    thread_id: str
    checkpoint_ns: Namespace

    @property
    def stored_ns(self) -> str:
        return encode_namespace(self.checkpoint_ns)

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
        checkpoint = dump_position(self.graph, position, stamp, self.stored_ns, step, source)
        await finish_despite_cancellation(self.checkpointer.aput(self.thread_id, checkpoint))
        return checkpoint

    async def begin(self, position: Position, step: int, parent_id: UUID | None) -> "ThreadJournal":
        checkpoint = await self.put(position, step, CheckpointSource.input, parent_id, parent_id)
        return ThreadJournal(self, CheckpointTuple(checkpoint=checkpoint, pending_writes=()), checkpoint.id)

    async def find(self, checkpoint_id: UUID | None) -> "Loaded | None":
        newest = await self.checkpointer.aget_tuple(self.thread_id, self.stored_ns, None)
        if checkpoint_id is None:
            return None if newest is None else Loaded(newest, newest.checkpoint.id)
        saved = await self.checkpointer.aget_tuple(self.thread_id, self.stored_ns, checkpoint_id)
        if saved is None or newest is None:
            raise LookupError(f"thread {self.thread_id!r} has no checkpoint {checkpoint_id}")
        return Loaded(saved, newest.checkpoint.id)

    async def open_loaded(self, loaded: "Loaded", *, fork: bool) -> "OpenedRun":
        saved = loaded.saved
        position = self.read_position(saved)
        if not fork:
            return OpenedRun(position, ThreadJournal(self, saved, loaded.newest_id))
        checkpoint = saved.checkpoint
        forked = await self.put(position, checkpoint.step, CheckpointSource.fork, checkpoint.id, loaded.newest_id)
        return OpenedRun(
            position, ThreadJournal(self, CheckpointTuple(checkpoint=forked, pending_writes=()), forked.id)
        )

    async def open_run(self, checkpoint_id: UUID | None) -> "OpenedRun | None":
        loaded = await self.find(checkpoint_id)
        return None if loaded is None else await self.open_loaded(loaded, fork=checkpoint_id is not None)

    async def receive(self, message: Message, fork_from: UUID | None) -> Start:
        loaded = await self.find(fork_from)
        if loaded is None:
            raise LookupError(f"thread {self.thread_id!r} has no checkpoint, so it accepts no message")
        level = (
            await self.read_level(loaded.saved, subgraphs=False) if fork_from is None else SavedLevel(loaded.saved, {})
        )
        accepts = self.build_snapshot(level, subgraphs=False).accepts
        match message.to:
            case None:
                match list(accepts):
                    case [entrance]:
                        pass
                    case []:
                        raise ValueError(f"thread {self.thread_id!r} accepts no message at this checkpoint")
                    case several:
                        raise ValueError(f"the thread accepts a message at each of {several!r}, so give Message.to")
            case Entrance() as entrance:
                if entrance not in accepts:
                    raise ValueError(
                        f"the thread accepts no message at {entrance!r}, it accepts one at {list(accepts)!r}"
                    )
            case _:
                assert_never(message.to)
        graph = find_graph(self.graph, entrance.checkpoint_ns)
        value = graph.receivers[entrance.node].adapter.validate_python(message.value)
        if entrance.checkpoint_ns == self.checkpoint_ns:
            opened = await self.open_loaded(loaded, fork=fork_from is not None)
            delivered = deliver(opened.position, graph.triggers, entrance.node, value)
            checkpoint = opened.journal.saved.checkpoint
            return FreshStart(delivered, await self.begin(delivered, checkpoint.step + 1, checkpoint.id))
        child = ThreadScope(graph, self.checkpointer, self.thread_id, entrance.checkpoint_ns)
        resting = await child.find(None)
        if resting is None:
            raise LookupError(f"the child graph at {entrance.checkpoint_ns!r} has no checkpoint")
        checkpoint = resting.saved.checkpoint
        delivered = deliver(child.read_position(resting.saved), graph.triggers, entrance.node, value)
        await child.begin(delivered, checkpoint.step + 1, checkpoint.id)
        return Continuation(self.read_position(loaded.saved), ThreadJournal(self, loaded.saved, loaded.newest_id))

    def find_child_scope(self, journal: "ThreadJournal", task: TaskRef) -> "ThreadScope | None":
        match task:
            case StartRef():
                return None
            case PullRef() | PushRef():
                action = self.graph.nodes[task.name].action
            case _:
                assert_never(task)
        match action:
            case CallPlan():
                return None
            case CompiledGraph() as child:
                return ThreadScope(child, self.checkpointer, self.thread_id, journal.make_task_ns(task))
            case _:
                assert_never(action)

    def plan_refs(self, saved: CheckpointTuple) -> list[TaskRef]:
        return [refer_task(task) for task in plan_tasks(self.read_position(saved), self.graph.triggers)]

    async def list_nodes_of_the_last_step(self, checkpoint: Checkpoint) -> set[str]:
        if checkpoint.source is not CheckpointSource.loop or checkpoint.parent_id is None:
            return set()
        before = await self.checkpointer.aget_tuple(self.thread_id, self.stored_ns, checkpoint.parent_id)
        if before is None:
            raise LookupError(f"thread {self.thread_id!r} has no checkpoint {checkpoint.parent_id}")
        nodes: set[str] = set()
        for task in self.plan_refs(before):
            match task:
                case StartRef():
                    pass
                case PullRef(name=name) | PushRef(name=name):
                    nodes.add(name)
                case _:
                    assert_never(task)
        return nodes

    async def read_level(self, saved: CheckpointTuple, *, subgraphs: bool) -> "SavedLevel":
        journal = ThreadJournal(self, saved, saved.checkpoint.id)
        children: dict[str, SavedLevel] = {}
        for task in self.plan_refs(saved):
            scope = self.find_child_scope(journal, task)
            if scope is None or not (subgraphs or journal.find_writes(task) is None):
                continue
            child = await self.checkpointer.aget_tuple(scope.thread_id, scope.stored_ns, None)
            if child is not None:
                children[journal.make_task_id(task)] = await scope.read_level(child, subgraphs=True)
        return SavedLevel(saved, children)

    def build_task_snapshot(
        self, journal: "ThreadJournal", task: TaskRef, level: "SavedLevel", *, subgraphs: bool
    ) -> TaskSnapshot:
        task_id = journal.make_task_id(task)
        writes = journal.find_writes(task)
        scope = self.find_child_scope(journal, task)
        child_level = level.children.get(task_id)
        child = None if scope is None or child_level is None else scope.build_snapshot(child_level, subgraphs=True)
        accepts: Mapping[Entrance, TypeForm[object]] = {} if child is None else child.accepts
        status: TaskStatus
        if writes is not None:
            status = FinishedTask(writes.fields)
        else:
            stop = journal.find_stop(task)
            match stop:
                case TaskFailed():
                    status = FailedTask(stop.error, accepts)
                case TaskWaiting() | None:
                    status = WaitingTask(accepts) if accepts else PendingTask()
                case _:
                    assert_never(stop)
        return TaskSnapshot(id=task_id, path=task, status=status, state=child if subgraphs else None)

    def build_snapshot(self, level: "SavedLevel", *, subgraphs: bool) -> StateSnapshot:
        saved = level.saved
        journal = ThreadJournal(self, saved, saved.checkpoint.id)
        position = self.read_position(saved)
        tasks = tuple(
            self.build_task_snapshot(journal, task, level, subgraphs=subgraphs) for task in self.plan_refs(saved)
        )
        waiting: dict[Entrance, TypeForm[object]] = {
            entrance: message_type
            for task in tasks
            if isinstance(task.status, WaitingTask | FailedTask)
            for entrance, message_type in task.status.accepts.items()
        }
        return StateSnapshot(
            values=position.channels.read_values(),
            checkpoint_ns=self.checkpoint_ns,
            checkpoint_id=saved.checkpoint.id,
            parent_id=saved.checkpoint.parent_id,
            created_at=saved.checkpoint.ts,
            step=saved.checkpoint.step,
            source=saved.checkpoint.source,
            tasks=tasks,
            accepts=waiting if tasks else self.graph.list_entrances(self.checkpoint_ns, position),
        )

    async def read_snapshot(self, saved: CheckpointTuple, *, subgraphs: bool) -> StateSnapshot:
        return self.build_snapshot(await self.read_level(saved, subgraphs=subgraphs), subgraphs=subgraphs)


@dataclass(frozen=True)
class SavedLevel:
    saved: CheckpointTuple
    children: Mapping[str, "SavedLevel"]


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
    def ns(self) -> Namespace:
        return self.scope.checkpoint_ns

    def make_task_id(self, task: TaskRef) -> str:
        checkpoint = self.saved.checkpoint
        return str(uuid5(checkpoint.id, f"{self.scope.stored_ns}|{checkpoint.step + 1}|{name_task(task)}"))

    async def save(self, position: Position, source: CheckpointSource) -> None:
        parent = self.saved.checkpoint
        checkpoint = await self.scope.put(position, parent.step + 1, source, parent.id, self.newest_id)
        self.saved = CheckpointTuple(checkpoint=checkpoint, pending_writes=())
        self.newest_id = checkpoint.id

    async def read_snapshot(self) -> StateSnapshot:
        return await self.scope.read_snapshot(self.saved, subgraphs=False)

    async def read_checkpoint(self) -> CheckpointPart:
        return CheckpointPart(await self.read_snapshot())

    def make_task_ns(self, task: TaskRef) -> Namespace:
        return (*self.scope.checkpoint_ns, Segment(task.name, self.make_task_id(task)))

    async def put_write(self, write: PendingWrite) -> None:
        scope = self.scope
        checkpoint = self.saved.checkpoint
        await finish_despite_cancellation(
            scope.checkpointer.aput_writes(scope.thread_id, scope.stored_ns, checkpoint.id, write)
        )
        kept = (kept for kept in self.saved.pending_writes if kept.key != write.key)
        self.saved = CheckpointTuple(checkpoint=checkpoint, pending_writes=(*kept, write))

    async def record(self, task: TaskRef, writes: TaskWrites) -> None:
        saved = dump_writes(self.scope.graph, writes)
        await self.put_write(TaskWritesRecord(task_id=self.make_task_id(task), writes=saved))

    async def record_failure(self, task: TaskRef, error: str) -> None:
        await self.put_write(TaskStopRecord(task_id=self.make_task_id(task), stop=TaskFailed(error=error)))

    async def record_waiting(self, task: TaskRef) -> None:
        await self.put_write(TaskStopRecord(task_id=self.make_task_id(task), stop=TaskWaiting()))

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

    async def record_command(self, writes: TaskWrites) -> None:
        index = sum(1 for write in self.saved.pending_writes if isinstance(write, RunCommandRecord))
        await self.put_write(RunCommandRecord(index=index, command=dump_writes(self.scope.graph, writes)))

    async def enter_child(self, child: AnyGraph, call: NodeCall) -> Start:
        scope = ThreadScope(child, self.scope.checkpointer, self.scope.thread_id, self.make_task_ns(call.task))
        opened = await scope.open_run(None)
        if opened is not None:
            return Continuation(opened.position, opened.journal)
        start = child.feed(child.open(extract_writes(call.node_input)), {})
        return FreshStart(start, await scope.begin(start, -1, None))


@dataclass(frozen=True)
class OpenedRun:
    position: Position
    journal: ThreadJournal


def infer_writer(
    graph: AnyGraph, versions_seen: Mapping[str, TriggerVersions], nodes_of_the_last_step: Collection[str]
) -> str:
    if len(graph.nodes) == 1:
        return next(iter(graph.nodes))
    if len(nodes_of_the_last_step) > 1:
        raise ValueError(
            f"the node that made the last update is not clear, because the nodes {sorted(nodes_of_the_last_step)!r} "
            "ran in the last step, so give as_node"
        )
    last_seen = sorted(
        (max(versions), node)
        for node, seen in versions_seen.items()
        if node in graph.nodes and (versions := [*seen.branches.values(), *(join.version for join in seen.joins)])
    )
    if len(last_seen) == 1 or (len(last_seen) > 1 and last_seen[-1][0] != last_seen[-2][0]):
        return last_seen[-1][1]
    raise ValueError("the node that made the last update is not clear, so give as_node")


def find_graph(graph: AnyGraph, checkpoint_ns: Namespace) -> AnyGraph:
    found = graph
    for segment in checkpoint_ns:
        name = segment.node
        match found.nodes[name].action if name in found.nodes else None:
            case CompiledGraph() as child:
                found = child
            case _:
                raise LookupError(f"the namespace {checkpoint_ns!r} names {name!r}, which is not a child graph")
    return found

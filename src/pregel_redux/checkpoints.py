import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


class CheckpointSource(Enum):
    """Names the action that saved a checkpoint.

    `input` marks the start of a run.
    `loop` marks a step of a run.
    `update` marks a call to `PersistentGraph.update_state`.
    `fork` marks a run that continues from `fork_from`.
    """

    input = "input"
    loop = "loop"
    update = "update"
    fork = "fork"


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")


class PendingSend(Record):
    node: str
    node_input: JsonValue


class SavedArrival(Record):
    sources: tuple[str, ...]
    target: str
    source: str


class SavedWrites(Record):
    fields: dict[str, JsonValue]
    branches: tuple[str, ...]
    arrivals: tuple[SavedArrival, ...]
    sends: tuple[PendingSend, ...]


class PendingKind(StrEnum):
    writes = "writes"
    stop = "stop"
    command = "command"


@dataclass(frozen=True, order=True)
class PendingKey:
    kind: PendingKind
    owner: str


class TaskWritesRecord(Record):
    kind: Literal[PendingKind.writes] = PendingKind.writes
    task_id: str
    writes: SavedWrites

    @property
    def key(self) -> PendingKey:
        return PendingKey(self.kind, self.task_id)


class StopKind(StrEnum):
    waiting = "waiting"
    failed = "failed"


class TaskWaiting(Record):
    outcome: Literal[StopKind.waiting] = StopKind.waiting


class TaskFailed(Record):
    outcome: Literal[StopKind.failed] = StopKind.failed
    error: str


class TaskStopRecord(Record):
    kind: Literal[PendingKind.stop] = PendingKind.stop
    task_id: str
    stop: Annotated[TaskWaiting | TaskFailed, Field(discriminator="outcome")]

    @property
    def key(self) -> PendingKey:
        return PendingKey(self.kind, self.task_id)


class RunCommandRecord(Record):
    kind: Literal[PendingKind.command] = PendingKind.command
    index: int
    command: SavedWrites

    @property
    def key(self) -> PendingKey:
        return PendingKey(self.kind, str(self.index))


type PendingWrite = Annotated[TaskWritesRecord | TaskStopRecord | RunCommandRecord, Field(discriminator="kind")]


class SavedBranch(Record):
    node: str
    finished: bool | None


class SavedJoin(Record):
    sources: tuple[str, ...]
    target: str
    seen: tuple[str, ...]
    finished: bool | None


class ChannelValues(Record):
    start: dict[str, JsonValue] | None
    fields: dict[str, JsonValue]
    branches: tuple[SavedBranch, ...]
    joins: tuple[SavedJoin, ...]
    tasks: tuple[PendingSend, ...]
    received: dict[str, JsonValue]


class JoinVersion(Record):
    sources: tuple[str, ...]
    target: str
    version: int


class TriggerVersions(Record):
    branches: dict[str, int]
    joins: tuple[JoinVersion, ...]


class ChannelVersions(Record):
    start: int | None
    fields: dict[str, int]
    triggers: TriggerVersions
    tasks: int | None


class Checkpoint(Record):
    """Holds the saved state of one graph at one step of a thread.

    A checkpointer stores the checkpoint as JSON.
    """

    v: Literal[2] = 2
    id: UUID
    parent_id: UUID | None
    ts: AwareDatetime
    checkpoint_ns: str
    step: int
    source: CheckpointSource
    channel_values: ChannelValues
    channel_versions: ChannelVersions
    versions_seen: dict[str, TriggerVersions]
    start_seen: int | None


class CheckpointTuple(Record):
    """Holds a checkpoint and the results of the tasks that ran after the checkpoint."""

    checkpoint: Checkpoint
    pending_writes: tuple[PendingWrite, ...]


class AsyncCheckpointer(Protocol):
    """Defines the asynchronous form of `Checkpointer`."""

    async def aput(self, thread_id: str, checkpoint: Checkpoint) -> None: ...
    async def aput_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite
    ) -> None: ...
    async def aget_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None
    ) -> CheckpointTuple | None: ...
    def alist(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> AsyncIterator[CheckpointTuple]: ...
    async def adelete_thread(self, thread_id: str) -> None: ...


class Checkpointer(Protocol):
    """Stores the checkpoints of each thread.

    The graph calls each method from a worker thread.
    Several calls run at the same time.
    """

    def put(self, thread_id: str, checkpoint: Checkpoint) -> None:
        """Stores `checkpoint` under `thread_id`.

        A checkpoint with the same id replaces the stored checkpoint.
        """
        ...

    def put_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        """Stores `write` for the checkpoint `checkpoint_id`.

        A write with the same `key` replaces the stored write.

        Raises
        ------
        KeyError
            The checkpoint `checkpoint_id` is absent.
        """
        ...

    def get_tuple(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None) -> CheckpointTuple | None:
        """Returns the checkpoint `checkpoint_id` with its writes.

        `checkpoint_id` of `None` selects the checkpoint with the greatest id.
        Returns `None` for an absent checkpoint.
        """
        ...

    def list(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> Iterator[CheckpointTuple]:
        """Yields the checkpoints of the graph `checkpoint_ns` in descending id order.

        Parameters
        ----------
        before
            Selects the checkpoints with an id less than `before`.
        limit
            The maximum number of checkpoints.
        """
        ...

    def delete_thread(self, thread_id: str) -> None:
        """Removes each checkpoint and each write of the thread."""
        ...


@dataclass(frozen=True)
class ThreadedCheckpointer:
    checkpointer: Checkpointer

    async def aput(self, thread_id: str, checkpoint: Checkpoint) -> None:
        await asyncio.to_thread(self.checkpointer.put, thread_id, checkpoint)

    async def aput_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        await asyncio.to_thread(self.checkpointer.put_writes, thread_id, checkpoint_ns, checkpoint_id, write)

    async def aget_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None
    ) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.checkpointer.get_tuple, thread_id, checkpoint_ns, checkpoint_id)

    async def alist(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> AsyncIterator[CheckpointTuple]:
        found = await asyncio.to_thread(
            lambda: tuple(self.checkpointer.list(thread_id, checkpoint_ns, before=before, limit=limit))
        )
        for saved in found:
            yield saved

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.checkpointer.delete_thread, thread_id)


@dataclass
class InMemoryCheckpointer:
    """Stores checkpoints as JSON in the memory of the process."""

    threads: dict[str, dict[str, dict[UUID, str]]] = field(default_factory=dict)
    pending: dict[tuple[str, str, UUID], dict[PendingKey, str]] = field(default_factory=dict)

    def put(self, thread_id: str, checkpoint: Checkpoint) -> None:
        records = self.threads.setdefault(thread_id, {}).setdefault(checkpoint.checkpoint_ns, {})
        records[checkpoint.id] = checkpoint.model_dump_json()

    def put_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        if checkpoint_id not in self.threads[thread_id][checkpoint_ns]:
            raise KeyError(checkpoint_id)
        records = self.pending.setdefault((thread_id, checkpoint_ns, checkpoint_id), {})
        records[write.key] = write.model_dump_json()

    def read_tuple(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID) -> CheckpointTuple:
        adapter = TypeAdapter[PendingWrite](PendingWrite)
        writes = sorted(self.pending.get((thread_id, checkpoint_ns, checkpoint_id), {}).items())
        return CheckpointTuple(
            checkpoint=Checkpoint.model_validate_json(self.threads[thread_id][checkpoint_ns][checkpoint_id]),
            pending_writes=tuple(adapter.validate_json(write) for _, write in writes),
        )

    def get_tuple(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None) -> CheckpointTuple | None:
        records = self.threads.get(thread_id, {}).get(checkpoint_ns, {})
        found = max(records, default=None) if checkpoint_id is None else checkpoint_id
        if found is None or found not in records:
            return None
        return self.read_tuple(thread_id, checkpoint_ns, found)

    def list(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> Iterator[CheckpointTuple]:
        records = self.threads.get(thread_id, {}).get(checkpoint_ns, {})
        found = sorted((key for key in records if before is None or key < before), reverse=True)
        for key in found[:limit]:
            yield self.read_tuple(thread_id, checkpoint_ns, key)

    def delete_thread(self, thread_id: str) -> None:
        self.threads.pop(thread_id, None)
        self.pending = {key: records for key, records in self.pending.items() if key[0] != thread_id}

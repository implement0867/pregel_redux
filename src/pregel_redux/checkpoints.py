import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue

from pregel_redux.interrupts import Interrupt


class CheckpointSource(Enum):
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
    interrupt = "interrupt"
    command = "command"


class TaskWritesRecord(Record):
    kind: Literal[PendingKind.writes] = PendingKind.writes
    task_id: str
    writes: SavedWrites


class TaskInterruptRecord(Record):
    kind: Literal[PendingKind.interrupt] = PendingKind.interrupt
    task_id: str
    interrupts: tuple[Interrupt, ...]
    resume: tuple[JsonValue, ...]


class RunCommandRecord(Record):
    kind: Literal[PendingKind.command] = PendingKind.command
    task_id: None = None
    commands: tuple[SavedWrites, ...]


type PendingWrite = Annotated[TaskWritesRecord | TaskInterruptRecord | RunCommandRecord, Field(discriminator="kind")]


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
    v: Literal[1] = 1
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
    checkpoint: Checkpoint
    pending_writes: tuple[PendingWrite, ...]


class AsyncCheckpointer(Protocol):
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


class Checkpointer(AsyncCheckpointer, Protocol):
    def put(self, thread_id: str, checkpoint: Checkpoint) -> None: ...
    def put_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None: ...
    def get_tuple(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None) -> CheckpointTuple | None: ...
    def list(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> Iterator[CheckpointTuple]: ...
    def delete_thread(self, thread_id: str) -> None: ...


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
    threads: dict[str, dict[str, dict[UUID, str]]] = field(default_factory=dict)

    def put(self, thread_id: str, checkpoint: Checkpoint) -> None:
        saved = CheckpointTuple(checkpoint=checkpoint, pending_writes=())
        self.threads.setdefault(thread_id, {}).setdefault(checkpoint.checkpoint_ns, {})[checkpoint.id] = (
            saved.model_dump_json()
        )

    def put_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        records = self.threads[thread_id][checkpoint_ns]
        saved = CheckpointTuple.model_validate_json(records[checkpoint_id])
        kept = tuple(item for item in saved.pending_writes if (item.kind, item.task_id) != (write.kind, write.task_id))
        records[checkpoint_id] = CheckpointTuple(
            checkpoint=saved.checkpoint, pending_writes=(*kept, write)
        ).model_dump_json()

    def get_tuple(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None) -> CheckpointTuple | None:
        records = self.threads.get(thread_id, {}).get(checkpoint_ns, {})
        found = max(records, default=None) if checkpoint_id is None else checkpoint_id
        if found is None or found not in records:
            return None
        return CheckpointTuple.model_validate_json(records[found])

    def list(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> Iterator[CheckpointTuple]:
        records = self.threads.get(thread_id, {}).get(checkpoint_ns, {})
        found = sorted((key for key in records if before is None or key < before), reverse=True)
        for key in found[:limit]:
            yield CheckpointTuple.model_validate_json(records[key])

    def delete_thread(self, thread_id: str) -> None:
        self.threads.pop(thread_id, None)

    async def aput(self, thread_id: str, checkpoint: Checkpoint) -> None:
        self.put(thread_id, checkpoint)

    async def aput_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        self.put_writes(thread_id, checkpoint_ns, checkpoint_id, write)

    async def aget_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None
    ) -> CheckpointTuple | None:
        return self.get_tuple(thread_id, checkpoint_ns, checkpoint_id)

    async def alist(
        self, thread_id: str, checkpoint_ns: str, *, before: UUID | None, limit: int | None
    ) -> AsyncIterator[CheckpointTuple]:
        for saved in self.list(thread_id, checkpoint_ns, before=before, limit=limit):
            yield saved

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

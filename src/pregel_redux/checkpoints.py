from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pregel_redux.interrupts import Interrupt


class CheckpointSource(Enum):
    input = "input"
    loop = "loop"


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
    resume = "resume"


class TaskWritesRecord(Record):
    kind: Literal[PendingKind.writes] = PendingKind.writes
    task_id: str
    writes: SavedWrites


class TaskInterruptRecord(Record):
    kind: Literal[PendingKind.interrupt] = PendingKind.interrupt
    task_id: str
    interrupts: tuple[Interrupt, ...]
    resume: tuple[JsonValue, ...]


class RunResumeRecord(Record):
    kind: Literal[PendingKind.resume] = PendingKind.resume
    task_id: None = None
    value: JsonValue


type PendingWrite = Annotated[TaskWritesRecord | TaskInterruptRecord | RunResumeRecord, Field(discriminator="kind")]


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
    id: UUID
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


class Checkpointer(Protocol):
    async def aput(self, thread_id: str, checkpoint: Checkpoint) -> None: ...
    async def aput_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite
    ) -> None: ...
    async def aget_tuple(self, thread_id: str, checkpoint_ns: str) -> CheckpointTuple | None: ...


@dataclass
class InMemoryCheckpointer:
    threads: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    async def aput(self, thread_id: str, checkpoint: Checkpoint) -> None:
        saved = CheckpointTuple(checkpoint=checkpoint, pending_writes=())
        self.threads.setdefault(thread_id, {}).setdefault(checkpoint.checkpoint_ns, []).append(saved.model_dump_json())

    async def aput_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        records = self.threads[thread_id][checkpoint_ns]
        saved = [CheckpointTuple.model_validate_json(record) for record in records]
        index = next(index for index, item in enumerate(saved) if item.checkpoint.id == checkpoint_id)
        kept = tuple(
            item for item in saved[index].pending_writes if (item.kind, item.task_id) != (write.kind, write.task_id)
        )
        records[index] = CheckpointTuple(
            checkpoint=saved[index].checkpoint, pending_writes=(*kept, write)
        ).model_dump_json()

    async def aget_tuple(self, thread_id: str, checkpoint_ns: str) -> CheckpointTuple | None:
        records = self.threads.get(thread_id, {}).get(checkpoint_ns)
        return CheckpointTuple.model_validate_json(records[-1]) if records else None

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, to_json


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, ser_json_inf_nan="constants")


class CheckpointSource(Enum):
    input = "input"
    loop = "loop"
    update = "update"
    fork = "fork"


class PlannedTaskKind(StrEnum):
    edge = "edge"
    send = "send"


class TaskStateKind(StrEnum):
    completed = "completed"
    interrupted = "interrupted"


class TaskAddress(Record):
    checkpoint_id: UUID
    index: int


class EdgeTask(Record):
    kind: Literal[PlannedTaskKind.edge] = PlannedTaskKind.edge
    node: str


class SendTask(Record):
    kind: Literal[PlannedTaskKind.send] = PlannedTaskKind.send
    node: str
    node_input: JsonValue


type PlannedTask = Annotated[EdgeTask | SendTask, Field(discriminator="kind")]


class Checkpoint(Record):
    thread_id: str
    caller: TaskAddress | None
    id: UUID
    parent_id: UUID | None
    created_at: AwareDatetime
    step: int
    source: CheckpointSource
    values: dict[str, JsonValue]
    tasks: tuple[PlannedTask, ...]


class CompletedTask(Record):
    kind: Literal[TaskStateKind.completed] = TaskStateKind.completed
    writes: dict[str, JsonValue]
    targets: tuple[str, ...]
    sends: tuple[SendTask, ...]


class InterruptedTask(Record):
    kind: Literal[TaskStateKind.interrupted] = TaskStateKind.interrupted
    answers: tuple[JsonValue, ...]
    payload: JsonValue


type TaskState = Annotated[CompletedTask | InterruptedTask, Field(discriminator="kind")]


class SavedCheckpoint(Record):
    checkpoint: Checkpoint
    task_states: dict[int, TaskState]


class Checkpointer(Protocol):
    async def put_checkpoint(self, checkpoint: Checkpoint) -> None: ...
    async def put_task_state(self, thread_id: str, task: TaskAddress, state: TaskState) -> None: ...
    async def read_checkpoint(self, thread_id: str, checkpoint_id: UUID) -> SavedCheckpoint | None: ...
    def list_checkpoints(self, thread_id: str, caller: TaskAddress | None) -> AsyncIterator[SavedCheckpoint]: ...
    async def delete_thread(self, thread_id: str) -> None: ...


@dataclass
class InMemoryCheckpointer:
    threads: dict[str, dict[UUID, str]] = field(default_factory=dict)

    async def put_checkpoint(self, checkpoint: Checkpoint) -> None:
        saved = SavedCheckpoint(checkpoint=checkpoint, task_states={})
        self.threads.setdefault(checkpoint.thread_id, {})[checkpoint.id] = saved.model_dump_json()

    async def put_task_state(self, thread_id: str, task: TaskAddress, state: TaskState) -> None:
        saved = await self.read_checkpoint(thread_id, task.checkpoint_id)
        if saved is None:
            raise LookupError(f"thread {thread_id!r} has no checkpoint {task.checkpoint_id}")
        updated = SavedCheckpoint(checkpoint=saved.checkpoint, task_states={**saved.task_states, task.index: state})
        self.threads[thread_id][task.checkpoint_id] = updated.model_dump_json()

    async def read_checkpoint(self, thread_id: str, checkpoint_id: UUID) -> SavedCheckpoint | None:
        record = self.threads.get(thread_id, {}).get(checkpoint_id)
        return None if record is None else SavedCheckpoint.model_validate_json(record)

    async def list_checkpoints(self, thread_id: str, caller: TaskAddress | None) -> AsyncIterator[SavedCheckpoint]:
        for record in reversed(self.threads.get(thread_id, {}).values()):
            saved = SavedCheckpoint.model_validate_json(record)
            if saved.checkpoint.caller == caller:
                yield saved

    async def delete_thread(self, thread_id: str) -> None:
        self.threads.pop(thread_id, None)


def load_value(adapter: TypeAdapter[object], data: JsonValue) -> object:
    return adapter.validate_json(to_json(data))


def dump_value(adapter: TypeAdapter[object], value: object, owner: str) -> JsonValue:
    try:
        data: JsonValue = adapter.dump_python(value, mode="json", by_alias=True, warnings="error")
        restored = load_value(adapter, data)
    except (PydanticSerializationError, ValidationError) as error:
        raise ValueError(f"{owner} holds {value!r}, which a checkpoint cannot store") from error
    if restored != value:
        raise ValueError(f"{owner} holds {value!r}, which a checkpoint restores as {restored!r}")
    return data

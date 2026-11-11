import dataclasses
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum, StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError


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
    node_input: str


type PlannedTask = Annotated[EdgeTask | SendTask, Field(discriminator="kind")]


class Checkpoint(Record):
    thread_id: str
    caller: TaskAddress | None
    id: UUID
    parent_id: UUID | None
    created_at: AwareDatetime
    step: int
    source: CheckpointSource
    values: dict[str, str]
    tasks: tuple[PlannedTask, ...]


class CompletedTask(Record):
    kind: Literal[TaskStateKind.completed] = TaskStateKind.completed
    writes: dict[str, str]
    targets: tuple[str, ...]
    sends: tuple[SendTask, ...]


class InterruptedTask(Record):
    kind: Literal[TaskStateKind.interrupted] = TaskStateKind.interrupted
    answers: tuple[str, ...]
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


def is_same_value(original: object, restored: object) -> bool:
    if type(original) is not type(restored) or original != restored:
        return False
    match original, restored:
        case BaseModel(), BaseModel():
            return original.model_fields_set == restored.model_fields_set and all(
                is_same_value(getattr(original, name), getattr(restored, name)) for name in type(original).model_fields
            )
        case datetime() | time(), _:
            return original.tzinfo is None or original.tzinfo.utcoffset(None) is not None
        case Mapping(), Mapping():
            original_entries: list[object] = [*original.keys(), *original.values()]
            restored_entries: list[object] = [*restored.keys(), *restored.values()]
            return all(map(is_same_value, original_entries, restored_entries))
        case list() | tuple(), list() | tuple():
            original_items: list[object] = [*original]
            restored_items: list[object] = [*restored]
            return all(map(is_same_value, original_items, restored_items))
        case set() | frozenset(), set() | frozenset():
            original_members: list[object] = [*original]
            restored_members: list[object] = [*restored]
            restored_by_value = {member: member for member in restored_members}
            return all(is_same_value(member, restored_by_value[member]) for member in original_members)
        case _ if dataclasses.is_dataclass(original):
            return all(
                is_same_value(getattr(original, field.name), getattr(restored, field.name))
                for field in dataclasses.fields(original)
            )
        case _:
            return True


def dump_value(adapter: TypeAdapter[object], value: object, owner: str) -> str:
    try:
        data = adapter.dump_json(value, by_alias=False, exclude_unset=True, round_trip=True, warnings="error").decode()
        restored = adapter.validate_json(data, by_name=True)
    except (PydanticSerializationError, ValidationError) as error:
        raise ValueError(f"{owner} holds {value!r}, which a checkpoint cannot store") from error
    if not is_same_value(value, restored):
        raise ValueError(f"{owner} holds {value!r}, which a checkpoint restores as {restored!r}")
    return data

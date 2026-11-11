from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from pydantic_core import to_json

from pregel_redux.channels import Channel, reject_unknown_channels


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")


class PendingSend(Record):
    node: str
    node_input: JsonValue


class JoinArrivals(Record):
    sources: tuple[str, ...]
    target: str
    arrived: tuple[str, ...]


class JoinVersion(Record):
    sources: tuple[str, ...]
    target: str
    version: int


class TriggerVersions(Record):
    branches: dict[str, int]
    joins: tuple[JoinVersion, ...]


class ChannelVersions(Record):
    fields: dict[str, int]
    triggers: TriggerVersions
    sends: int | None


class TaskAddress(Record):
    checkpoint_id: UUID
    node: str
    send_index: int | None


class Checkpoint(Record):
    id: UUID
    parent: TaskAddress | None
    values: dict[str, JsonValue]
    branches: tuple[str, ...]
    joins: tuple[JoinArrivals, ...]
    sends: tuple[PendingSend, ...]
    versions: ChannelVersions
    seen: dict[str, TriggerVersions]


class Checkpointer(Protocol):
    async def put(self, thread_id: str, checkpoint: Checkpoint) -> None: ...
    async def get_latest(self, thread_id: str, parent: TaskAddress | None) -> Checkpoint | None: ...


@dataclass
class InMemoryCheckpointer:
    threads: dict[str, list[str]] = field(default_factory=dict)

    async def put(self, thread_id: str, checkpoint: Checkpoint) -> None:
        self.threads.setdefault(thread_id, []).append(checkpoint.model_dump_json())

    async def get_latest(self, thread_id: str, parent: TaskAddress | None) -> Checkpoint | None:
        checkpoints = (Checkpoint.model_validate_json(record) for record in reversed(self.threads.get(thread_id, [])))
        return next((checkpoint for checkpoint in checkpoints if checkpoint.parent == parent), None)


def dump_value(adapter: TypeAdapter[object], value: object) -> JsonValue:
    return adapter.dump_python(value, mode="json", by_alias=False, round_trip=True)


def load_value(adapter: TypeAdapter[object], value: JsonValue) -> object:
    return adapter.validate_json(to_json(value, inf_nan_mode="strings"), by_alias=False, by_name=True)


def dump_values(channels: Mapping[str, Channel], values: Mapping[str, object]) -> dict[str, JsonValue]:
    return {name: dump_value(channels[name].adapter, value) for name, value in values.items()}


def load_values(channels: Mapping[str, Channel], values: Mapping[str, JsonValue]) -> dict[str, object]:
    reject_unknown_channels(channels, values)
    return {name: load_value(channels[name].adapter, value) for name, value in values.items()}

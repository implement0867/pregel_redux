from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, JsonValue

from pregel_redux.channels import Channel


class PendingSend(BaseModel):
    model_config = ConfigDict(frozen=True)

    node: str
    values: dict[str, JsonValue]


class Checkpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    values: dict[str, JsonValue]
    nodes: tuple[str, ...]
    sends: tuple[PendingSend, ...]


def dump_values(channels: Mapping[str, Channel], values: Mapping[str, object]) -> dict[str, JsonValue]:
    return {
        name: channels[name].adapter.dump_python(value, mode="json", by_alias=True, round_trip=True)
        for name, value in values.items()
    }

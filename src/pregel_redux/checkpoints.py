from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, JsonValue
from pydantic_core import to_json

from pregel_redux.channels import Channel, reject_unknown_channels


class PendingSend(BaseModel):
    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    node: str
    values: dict[str, JsonValue]


class Checkpoint(BaseModel):
    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    values: dict[str, JsonValue]
    nodes: tuple[str, ...]
    sends: tuple[PendingSend, ...]


def dump_values(channels: Mapping[str, Channel], values: Mapping[str, object]) -> dict[str, JsonValue]:
    return {
        name: channels[name].adapter.dump_python(value, mode="json", by_alias=False, round_trip=True)
        for name, value in values.items()
    }


def load_values(channels: Mapping[str, Channel], values: Mapping[str, JsonValue]) -> dict[str, object]:
    reject_unknown_channels(channels, values)
    return {
        name: channels[name].adapter.validate_json(to_json(value, inf_nan_mode="strings"), by_alias=False, by_name=True)
        for name, value in values.items()
    }

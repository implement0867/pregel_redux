from dataclasses import dataclass
from operator import add, mul
from typing import Annotated, NotRequired, TypedDict

import pytest
from pydantic import BaseModel

from pregel_redux.channels import Channel, apply_writes, channels_from_schema, merge_channels, writes_from_update


class DictState[T](TypedDict):
    items: Annotated[list[T], add]
    latest: NotRequired[T]


@dataclass
class DataclassState[T]:
    items: Annotated[list[T], add]
    latest: T


class ModelState[T](BaseModel):
    items: Annotated[list[T], add]
    latest: T


class OtherReducerState(TypedDict):
    items: Annotated[list[int], mul]


@pytest.mark.parametrize("schema", [DictState[int], DataclassState[int], ModelState[int]])
def test_parameterized_generic_schema_gives_one_typed_channel_per_field(schema: object) -> None:
    assert channels_from_schema(schema) == {
        "items": Channel("items", list[int], add),
        "latest": Channel("latest", int, None),
    }


def test_unparameterized_generic_schema_keeps_its_type_parameter() -> None:
    (parameter,) = DictState.__type_params__

    assert channels_from_schema(DictState)["items"] == Channel("items", list[parameter], add)


def test_schema_that_has_no_fields_is_rejected() -> None:
    with pytest.raises(TypeError, match="not a TypedDict"):
        channels_from_schema(list[int])


def test_merged_schemas_reject_two_reducers_for_one_channel() -> None:
    with pytest.raises(ValueError, match="two different reducers"):
        merge_channels([DictState[int], OtherReducerState])


def test_writes_come_from_the_fields_that_a_model_update_sets() -> None:
    class Update(BaseModel):
        items: list[int] = []
        latest: int = 0

    assert writes_from_update(Update(latest=3)) == {"latest": 3}


def test_update_of_an_unsupported_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="not a mapping"):
        writes_from_update(3)


def test_reducer_combines_the_writes_of_one_step() -> None:
    channels = channels_from_schema(DictState[int])

    assert apply_writes(channels, {"items": [0]}, [{"items": [1]}, {"items": [2]}]) == {"items": [0, 1, 2]}


def test_channel_without_reducer_rejects_two_writes_in_one_step() -> None:
    channels = channels_from_schema(DictState[int])

    with pytest.raises(ValueError, match="more than one value"):
        apply_writes(channels, {}, [{"latest": 1}, {"latest": 2}])


def test_write_to_an_unknown_channel_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a channel"):
        apply_writes(channels_from_schema(DictState[int]), {}, [{"missing": 1}])

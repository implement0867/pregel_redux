from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import add, mul
from typing import Annotated, NotRequired, TypedDict

import pytest
from pydantic import BaseModel

from pregel_redux.channels import Channel, apply_writes, build_channels, extract_writes, merge_channels


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


class ModelUpdate(BaseModel):
    items: list[int] = []
    latest: int = 0


@dataclass
class DataclassUpdate:
    latest: int


@pytest.fixture
def channels() -> dict[str, Channel]:
    return build_channels(DictState[int])


@pytest.mark.parametrize(
    ("schema", "argument"),
    [
        (DictState[int], int),
        (DataclassState[int], int),
        (ModelState[int], int),
        (DictState[str], str),
        (DataclassState[list[int]], list[int]),
    ],
    ids=["typed-dict", "dataclass", "model", "typed-dict-of-str", "dataclass-of-list"],
)
def test_parameterized_generic_schema_gives_one_typed_channel_per_field(schema: object, argument: object) -> None:
    expected = {
        "items": Channel("items", list[argument], add),
        "latest": Channel("latest", argument, None),
    }

    actual = build_channels(schema)

    assert actual == expected, f"Expected {expected}, got {actual}. Each field must keep its type argument."


def test_unparameterized_generic_schema_keeps_its_type_parameter() -> None:
    (parameter,) = DictState.__type_params__
    expected = Channel("items", list[parameter], add)

    actual = build_channels(DictState)["items"]

    assert actual == expected, f"Expected {expected}, got {actual}. The type parameter must stay in the value type."


@pytest.mark.parametrize("schema", [list[int], int, "State", None], ids=repr)
def test_schema_that_has_no_fields_is_rejected(schema: object) -> None:
    with pytest.raises(TypeError):
        build_channels(schema)


def test_merged_schemas_reject_two_reducers_for_one_channel() -> None:
    with pytest.raises(ValueError):
        merge_channels([DictState[int], OtherReducerState])


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        (None, {}),
        ({"latest": 3}, {"latest": 3}),
        (ModelUpdate(latest=3), {"latest": 3}),
        (ModelUpdate(), {}),
        (DataclassUpdate(latest=3), {"latest": 3}),
    ],
    ids=["none", "mapping", "model-with-one-set-field", "model-with-no-set-field", "dataclass"],
)
def test_writes_come_from_the_fields_that_an_update_sets(update: object, expected: Mapping[str, object]) -> None:
    actual = extract_writes(update)

    assert actual == expected, f"Expected {expected}, got {actual}. Only the fields that the update sets are writes."


@pytest.mark.parametrize("update", [3, "text", [1], DataclassUpdate], ids=repr)
def test_update_of_an_unsupported_type_is_rejected(update: object) -> None:
    with pytest.raises(TypeError):
        extract_writes(update)


@pytest.mark.parametrize(
    ("values", "updates", "expected"),
    [
        ({"items": [0]}, [{"items": [1]}, {"items": [2]}], {"items": [0, 1, 2]}),
        ({}, [{"items": [1]}, {"items": [2]}], {"items": [1, 2]}),
        ({"latest": 1}, [{"latest": 2}], {"latest": 2}),
        ({"items": [0], "latest": 1}, [{"items": [1]}, {"latest": 2}], {"items": [0, 1], "latest": 2}),
        ({"items": [0]}, [], {"items": [0]}),
    ],
    ids=["reducer-with-value", "reducer-without-value", "replacement", "two-channels", "no-updates"],
)
def test_writes_of_one_step_give_the_next_values(
    channels: dict[str, Channel],
    values: Mapping[str, object],
    updates: Sequence[Mapping[str, object]],
    expected: Mapping[str, object],
) -> None:
    actual = apply_writes(channels, values, updates)

    assert actual == expected, f"Expected {expected}, got {actual}. The reducer must combine the writes of one step."


@pytest.mark.parametrize(
    "updates",
    [[{"latest": 1}, {"latest": 2}], [{"missing": 1}]],
    ids=["two-writes-without-reducer", "unknown-channel"],
)
def test_invalid_writes_are_rejected(channels: dict[str, Channel], updates: Sequence[Mapping[str, object]]) -> None:
    with pytest.raises(ValueError):
        apply_writes(channels, {}, updates)

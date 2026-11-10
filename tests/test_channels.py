from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, make_dataclass
from operator import add, mul
from types import new_class
from typing import Annotated, NotRequired, TypedDict

import pytest
from pydantic import BaseModel

from pregel_redux import Depends
from pregel_redux.channels import (
    Channel,
    Reducer,
    apply_writes,
    build_channels,
    extract_writes,
    merge_channels,
)


class DictState[T](TypedDict):
    items: Annotated[list[T], Reducer(add)]
    latest: NotRequired[T]


@dataclass
class DataclassState[T]:
    items: Annotated[list[T], Reducer(add)]
    latest: T


class ModelState[T](BaseModel):
    items: Annotated[list[T], Reducer(add)]
    latest: T


class OtherReducerState(TypedDict):
    items: Annotated[list[int], Reducer(mul)]


class ModelUpdate(BaseModel):
    items: list[int] = []
    latest: int = 0


@dataclass
class DataclassUpdate:
    latest: int


@pytest.fixture
def channels() -> dict[str, Channel]:
    return build_channels(DictState[int])


@pytest.fixture(params=[(TypedDict,), (BaseModel,), ()], ids=["typed-dict", "model", "dataclass"])
def build_schema(request: pytest.FixtureRequest) -> Callable[[object], object]:
    bases: tuple[object, ...] = request.param

    def build_class(hint: object) -> object:
        if not bases:
            return make_dataclass("Schema", [("items", hint)])
        return new_class("Schema", bases, exec_body=lambda namespace: namespace.update(__annotations__={"items": hint}))

    return build_class


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


@pytest.mark.parametrize(
    "metadata",
    [(Reducer(add),), (Reducer(add), "note"), ("note", Reducer(add)), ("note", Reducer(add), mul)],
    ids=["marker-alone", "marker-first", "marker-last", "marker-in-the-middle"],
)
def test_reducer_marker_at_any_position_gives_the_reducer_of_the_channel(
    build_schema: Callable[[object], object], metadata: tuple[object, ...]
) -> None:
    schema = build_schema(Annotated[list[int], *metadata])
    expected = Channel("items", list[int], add)

    actual = build_channels(schema)["items"]

    assert actual == expected, f"Expected {expected}, got {actual}. The Reducer marker must give the reducer."


@pytest.mark.parametrize(
    "metadata",
    [(add,), (mul,), ("note", add), (Depends(list),)],
    ids=["add", "mul", "callable-in-last-position", "other-marker"],
)
def test_annotation_without_a_reducer_marker_gives_a_channel_without_a_reducer(
    build_schema: Callable[[object], object], metadata: tuple[object, ...]
) -> None:
    schema = build_schema(Annotated[list[int], *metadata])
    expected = Channel("items", list[int], None)

    actual = build_channels(schema)["items"]

    assert actual == expected, f"Expected {expected}, got {actual}. Only a Reducer marker declares a reducer."


@pytest.mark.parametrize(
    "metadata",
    [(Reducer(add), Reducer(add)), (Reducer(add), Reducer(mul)), (Reducer(add), "note", Reducer(mul))],
    ids=["same-reducer-two-times", "two-reducers", "two-reducers-apart"],
)
def test_field_with_more_than_one_reducer_marker_is_rejected(
    build_schema: Callable[[object], object], metadata: tuple[object, ...]
) -> None:
    schema = build_schema(Annotated[list[int], *metadata])

    with pytest.raises(TypeError):
        build_channels(schema)


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

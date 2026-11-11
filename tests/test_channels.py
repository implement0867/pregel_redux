from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, make_dataclass
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
    build_start_values,
    extract_writes,
    merge_channels,
)
from tests.conftest import append_item, join_lists


class DictState[T](TypedDict):
    items: Annotated[list[T], Reducer(join_lists, initial_factory=list)]
    latest: NotRequired[T]


@dataclass
class DataclassState[T]:
    items: Annotated[list[T], Reducer(join_lists, initial_factory=list)]
    latest: T


class ModelState[T](BaseModel):
    items: Annotated[list[T], Reducer(join_lists, initial_factory=list)]
    latest: T


class AppendState(TypedDict):
    items: Annotated[list[int], Reducer(append_item, initial_factory=list)]


class OtherInitialFactoryState(TypedDict):
    items: Annotated[list[int], Reducer(join_lists, initial_factory=lambda: [0])]


class InitialValueState(TypedDict):
    items: Annotated[list[int], Reducer(join_lists, initial=[])]


class ModelUpdate(BaseModel):
    items: list[int] = []
    latest: int = 0


@dataclass
class DataclassUpdate:
    latest: int


@dataclass
class DataclassUpdateWithDefaults:
    items: list[int] = field(default_factory=list)
    latest: int = 0


def build_marker() -> Reducer[list[int], list[int]]:
    return Reducer(join_lists, initial_factory=list)


def build_append_marker() -> Reducer[list[int], int]:
    return Reducer(append_item, initial_factory=list)


def call_without_type_check(function: Callable[..., object], arguments: Mapping[str, object]) -> object:
    return function(join_lists, **arguments)


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
    """
    Arrange: a generic schema of each kind with a type argument.
    Act: build the channels of the schema.
    Assert: each channel has the type argument in its value type, and the reducer field has its marker.
    """
    expected = {
        "items": Channel("items", list[argument], build_marker()),
        "latest": Channel("latest", argument, None),
    }

    actual = build_channels(schema)

    assert actual == expected, f"Expected {expected}, got {actual}. Each field must keep its type argument."


def test_unparameterized_generic_schema_keeps_its_type_parameter() -> None:
    """
    Arrange: a generic schema without a type argument.
    Act: build the channels of the schema.
    Assert: the value type of the channel contains the type parameter.
    """
    (parameter,) = DictState.__type_params__
    expected = Channel("items", list[parameter], build_marker())

    actual = build_channels(DictState)["items"]

    assert actual == expected, f"Expected {expected}, got {actual}. The type parameter must stay in the value type."


@pytest.mark.parametrize(
    "metadata",
    [(build_marker(),), (build_marker(), "note"), ("note", build_marker()), ("note", build_marker(), mul)],
    ids=["marker-alone", "marker-first", "marker-last", "marker-in-the-middle"],
)
def test_reducer_marker_at_any_position_gives_the_reducer_of_the_channel(
    build_schema: Callable[[object], object], metadata: tuple[object, ...]
) -> None:
    """
    Arrange: a schema of each kind with a field that has one Reducer marker among other annotations.
    Act: build the channels of the schema.
    Assert: the channel has the marker as its reducer.
    """
    schema = build_schema(Annotated[list[int], *metadata])
    expected = Channel("items", list[int], build_marker())

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
    """
    Arrange: a schema of each kind with a field that has annotations but no Reducer marker.
    Act: build the channels of the schema.
    Assert: the channel has no reducer.
    """
    schema = build_schema(Annotated[list[int], *metadata])
    expected = Channel("items", list[int], None)

    actual = build_channels(schema)["items"]

    assert actual == expected, f"Expected {expected}, got {actual}. Only a Reducer marker declares a reducer."


@pytest.mark.parametrize(
    "metadata",
    [
        (build_marker(), build_marker()),
        (build_marker(), build_append_marker()),
        (build_marker(), "note", build_append_marker()),
    ],
    ids=["same-reducer-two-times", "two-reducers", "two-reducers-apart"],
)
def test_field_with_more_than_one_reducer_marker_is_rejected(
    build_schema: Callable[[object], object], metadata: tuple[object, ...]
) -> None:
    """
    Arrange: a schema of each kind with a field that has two Reducer markers.
    Act: build the channels of the schema.
    Assert: a TypeError is raised.
    """
    schema = build_schema(Annotated[list[int], *metadata])

    with pytest.raises(TypeError):
        build_channels(schema)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        (Reducer(join_lists, initial=[1]), [1]),
        (Reducer(join_lists, initial_factory=lambda: [1]), [1]),
        (Reducer(max, initial=0), 0),
        (build_append_marker(), []),
    ],
    ids=["initial-list", "factory-of-a-list", "initial-number", "factory-of-an-empty-list"],
)
def test_reducer_marker_gives_its_declared_starting_value(marker: Reducer[object, object], expected: object) -> None:
    """
    Arrange: a Reducer marker with an initial value or an initial factory.
    Act: make the starting value.
    Assert: the starting value is the declared one.
    """
    actual = marker.make_initial()

    assert actual == expected, f"Expected {expected}, got {actual}. The marker must give its declared starting value."


@pytest.mark.parametrize(
    "marker",
    [Reducer(join_lists, initial=[1]), Reducer(join_lists, initial_factory=lambda: [1])],
    ids=["initial", "initial-factory"],
)
def test_reducer_marker_gives_a_new_starting_value_each_time(marker: Reducer[list[int], list[int]]) -> None:
    """
    Arrange: a Reducer marker with a mutable starting value.
    Act: make the starting value two times.
    Assert: the two values are different objects.
    """
    first = marker.make_initial()

    second = marker.make_initial()

    assert first is not second, "Two runs must not share one mutable starting value."


@pytest.mark.parametrize(
    "arguments",
    [{}, {"initial": [], "initial_factory": list}],
    ids=["no-starting-value", "two-starting-values"],
)
def test_reducer_marker_without_exactly_one_starting_value_is_rejected(arguments: Mapping[str, object]) -> None:
    """
    Arrange: the arguments of a Reducer marker with no starting value or with two.
    Act: make the marker in a way that the type checker cannot examine.
    Assert: a TypeError is raised.
    """
    with pytest.raises(TypeError):
        call_without_type_check(Reducer, arguments)


@pytest.mark.parametrize("schema", [list[int], int, "State", None], ids=repr)
def test_schema_that_has_no_fields_is_rejected(schema: object) -> None:
    with pytest.raises(TypeError):
        build_channels(schema)


@pytest.mark.parametrize(
    "other_schema",
    [AppendState, OtherInitialFactoryState, InitialValueState],
    ids=["other-function", "other-initial-factory", "initial-value-and-not-a-factory"],
)
def test_merged_schemas_reject_two_reducers_for_one_channel(other_schema: object) -> None:
    """
    Arrange: two schemas that declare one channel with reducers that differ in function or in starting value.
    Act: merge the channels of the schemas.
    Assert: a ValueError is raised.
    """
    with pytest.raises(ValueError):
        merge_channels([DictState[int], other_schema])


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        (None, {}),
        ({"latest": 3}, {"latest": 3}),
        (ModelUpdate(latest=3), {"latest": 3}),
        (ModelUpdate(), {}),
        (DataclassUpdate(latest=3), {"latest": 3}),
        (DataclassUpdateWithDefaults(), {}),
        (DataclassUpdateWithDefaults(latest=3), {"latest": 3}),
        (DataclassUpdateWithDefaults(items=[1]), {"items": [1]}),
        (DataclassUpdateWithDefaults(items=[], latest=0), {}),
    ],
    ids=[
        "none",
        "mapping",
        "model-with-one-set-field",
        "model-with-no-set-field",
        "dataclass",
        "dataclass-with-only-defaults",
        "dataclass-with-one-value-that-is-not-the-default",
        "dataclass-with-one-factory-value-that-is-not-the-default",
        "dataclass-with-values-equal-to-the-defaults",
    ],
)
def test_writes_come_from_the_fields_that_an_update_sets(update: object, expected: Mapping[str, object]) -> None:
    """
    Arrange: an update of each supported kind.
    Act: extract the writes of the update.
    Assert: the writes are the fields that the update sets to a value other than the declared default.
    """
    actual = extract_writes(update)

    assert actual == expected, f"Expected {expected}, got {actual}. Only the fields that the update sets are writes."


@pytest.mark.parametrize("update", [3, "text", [1], DataclassUpdate], ids=repr)
def test_update_of_an_unsupported_type_is_rejected(update: object) -> None:
    with pytest.raises(TypeError):
        extract_writes(update)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({}, {"items": []}),
        ({"latest": 1}, {"items": [], "latest": 1}),
        ({"items": [5]}, {"items": [5]}),
        ({"items": [5], "latest": 1}, {"items": [5], "latest": 1}),
    ],
    ids=["nothing-given", "plain-channel-given", "reducer-channel-given", "both-given"],
)
def test_start_values_are_the_initial_values_and_the_given_values(
    channels: dict[str, Channel], given: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    """
    Arrange: the channels of a schema with one reducer field and one plain field, and some given values.
    Act: build the start values.
    Assert: each reducer channel has its initial value unless a value is given, and a given value is kept as it is.
    """
    actual = build_start_values(channels, given)

    assert actual == expected, f"Expected {expected}, got {actual}. A given value must replace the initial value."


def test_start_values_reject_a_given_value_for_an_unknown_channel(channels: dict[str, Channel]) -> None:
    """
    Arrange: the channels of a schema and a given value for a name that is not a channel.
    Act: build the start values.
    Assert: a ValueError is raised.
    """
    with pytest.raises(ValueError):
        build_start_values(channels, {"missing": 1})


@pytest.mark.parametrize(
    ("given", "updates", "expected"),
    [
        ({"items": [0]}, [{"items": [1]}, {"items": [2]}], {"items": [0, 1, 2]}),
        ({}, [{"items": [1]}, {"items": [2]}], {"items": [1, 2]}),
        ({"latest": 1}, [{"latest": 2}], {"items": [], "latest": 2}),
        ({"items": [0], "latest": 1}, [{"items": [1]}, {"latest": 2}], {"items": [0, 1], "latest": 2}),
        ({"items": [0]}, [], {"items": [0]}),
    ],
    ids=["reducer-with-given-value", "reducer-with-initial-value", "replacement", "two-channels", "no-updates"],
)
def test_writes_of_one_step_give_the_next_values(
    channels: dict[str, Channel],
    given: Mapping[str, object],
    updates: Sequence[Mapping[str, object]],
    expected: Mapping[str, object],
) -> None:
    """
    Arrange: the start values of a schema with one reducer field and one plain field, and the updates of one step.
    Act: apply the updates.
    Assert: the reducer combined the writes to its channel, and a write to the plain channel replaced the value.
    """
    actual = apply_writes(channels, build_start_values(channels, given), updates)

    assert actual == expected, f"Expected {expected}, got {actual}. The reducer must combine the writes of one step."


@pytest.mark.parametrize(
    ("updates", "expected"),
    [([{"items": 1}], [1]), ([{"items": 1}, {"items": 2}], [1, 2]), ([{"items": [1]}], [[1]])],
    ids=["one-write", "two-writes", "write-of-a-list"],
)
def test_first_write_to_a_reducer_channel_goes_through_the_reducer(
    updates: Sequence[Mapping[str, object]], expected: Sequence[object]
) -> None:
    """
    Arrange: the start values of a schema with a reducer whose update type is not its channel type.
    Act: apply the first writes to the channel.
    Assert: the reducer combined each write with the initial value, so the channel holds a list.
    """
    channels = build_channels(AppendState)

    actual = apply_writes(channels, build_start_values(channels, {}), updates)["items"]

    assert actual == expected, f"Expected {expected}, got {actual}. The first write must not replace the initial value."


@pytest.mark.parametrize(
    "updates",
    [[{"latest": 1}, {"latest": 2}], [{"missing": 1}]],
    ids=["two-writes-without-reducer", "unknown-channel"],
)
def test_invalid_writes_are_rejected(channels: dict[str, Channel], updates: Sequence[Mapping[str, object]]) -> None:
    """
    Arrange: updates with two writes to a channel without a reducer, or a write to an unknown channel.
    Act: apply the updates.
    Assert: a ValueError is raised.
    """
    with pytest.raises(ValueError):
        apply_writes(channels, build_start_values(channels, {}), updates)

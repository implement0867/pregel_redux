from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass, field, make_dataclass
from operator import add, mul
from types import new_class
from typing import Annotated, NotRequired, TypedDict

import pytest
from pydantic import BaseModel, ConfigDict, Field, PydanticSchemaGenerationError

from pregel_redux import Depends
from pregel_redux.channels import (
    Channel,
    Reducer,
    build_channels,
    extract_writes,
    merge_channels,
    parse_updates,
    reject_unknown_channels,
)
from pregel_redux.pregel import TaskWrites, apply_writes, open_channels
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
    items: Annotated[list[object], Reducer(append_item, initial_factory=list, update_type=object)]


class Total(BaseModel):
    count: int


class TotalState(TypedDict):
    total: Total


class Connection:
    pass


class ConnectionDict(TypedDict):
    connections: list[Connection]


@dataclass
class ConnectionDataclass:
    connection: Connection


class ConnectionModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: Connection


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
    return Reducer(append_item, initial_factory=list, update_type=int)


def apply_updates(
    channels: Mapping[str, Channel], values: Mapping[str, object], updates: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    writes = [TaskWrites(parse_updates(channels, update), frozenset(), (), ()) for update in updates]
    return apply_writes(open_channels(channels, values, {}, {}), [], writes, [], []).channels.read_values()


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
        "items": (list[argument], build_marker()),
        "latest": (argument, None),
    }

    actual = {name: (channel.value_type, channel.reducer) for name, channel in build_channels(schema).items()}

    assert actual == expected, f"Expected {expected}, got {actual}. Each field must keep its type argument."


def test_unparameterized_generic_schema_keeps_its_type_parameter() -> None:
    """
    Arrange: a generic schema without a type argument.
    Act: build the channels of the schema.
    Assert: the value type of the channel contains the type parameter.
    """
    (parameter,) = DictState.__type_params__
    expected = (list[parameter], build_marker())
    channel = build_channels(DictState)["items"]

    actual = (channel.value_type, channel.reducer)

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
    expected = (list[int], build_marker())
    channel = build_channels(schema)["items"]

    actual = (channel.value_type, channel.reducer)

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
    expected = (list[int], None)
    channel = build_channels(schema)["items"]

    actual = (channel.value_type, channel.reducer)

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
    ("names", "expectation"),
    [
        ([], does_not_raise()),
        (["items"], does_not_raise()),
        (["items", "latest"], does_not_raise()),
        (["missing"], pytest.raises(ValueError)),
        (["items", "missing"], pytest.raises(ValueError)),
    ],
    ids=["no-names", "one-channel", "all-channels", "unknown-name", "channel-and-unknown-name"],
)
def test_names_that_are_not_channels_are_rejected(
    channels: dict[str, Channel], names: Sequence[str], expectation: AbstractContextManager[object]
) -> None:
    """
    Arrange: the channels of a schema and a list of names.
    Act: check the names against the channels.
    Assert: a ValueError is raised only when one of the names is not a channel.
    """
    with expectation:
        reject_unknown_channels(channels, names)


@pytest.mark.parametrize(
    ("values", "updates", "expected"),
    [
        ({"items": [0]}, [{"items": [1]}, {"items": [2]}], {"items": [0, 1, 2]}),
        ({"items": []}, [{"items": [1]}, {"items": [2]}], {"items": [1, 2]}),
        ({"items": [], "latest": 1}, [{"latest": 2}], {"items": [], "latest": 2}),
        ({"items": [0], "latest": 1}, [{"items": [1]}, {"latest": 2}], {"items": [0, 1], "latest": 2}),
        ({"items": [0]}, [], {"items": [0]}),
    ],
    ids=["reducer-with-a-value", "reducer-with-the-initial-value", "replacement", "two-channels", "no-updates"],
)
def test_writes_of_one_step_give_the_next_values(
    channels: dict[str, Channel],
    values: Mapping[str, object],
    updates: Sequence[Mapping[str, object]],
    expected: Mapping[str, object],
) -> None:
    """
    Arrange: the values of a schema with one reducer field and one plain field, and the updates of one step.
    Act: apply the updates.
    Assert: the reducer combined the writes to its channel, and a write to the plain channel replaced the value.
    """
    actual = apply_updates(channels, values, updates)

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
    Arrange: the initial value of a channel with a reducer whose update type is not its channel type.
    Act: apply the first writes to the channel.
    Assert: the reducer combined each write with the initial value, so the channel holds a list.
    """
    channels = build_channels(AppendState)

    actual = apply_updates(channels, {"items": []}, updates)["items"]

    assert actual == expected, f"Expected {expected}, got {actual}. The first write must not replace the initial value."


@pytest.mark.parametrize(
    "updates",
    [
        [{"latest": 1}, {"latest": 2}],
        [{"missing": 1}],
        [{"latest": "text"}],
        [{"latest": None}],
        [{"items": ["text"]}],
        [{"items": [1]}, {"items": ["text"]}],
    ],
    ids=[
        "two-writes-without-reducer",
        "unknown-channel",
        "value-of-another-type",
        "none",
        "reducer-result-of-another-type",
        "second-reducer-result-of-another-type",
    ],
)
def test_invalid_writes_are_rejected(channels: dict[str, Channel], updates: Sequence[Mapping[str, object]]) -> None:
    """
    Arrange: updates with two writes to a channel without a reducer, a write to an unknown channel,
        or a write that gives a value that is not of the declared type of the channel.
    Act: apply the updates.
    Assert: a ValueError is raised.
    """
    with pytest.raises(ValueError):
        apply_updates(channels, {"items": []}, updates)


@pytest.mark.parametrize(
    "written",
    [Total(count=1), {"count": 1}, {"count": "1"}],
    ids=["model-instance", "mapping-of-the-fields", "mapping-with-a-number-as-text"],
)
def test_written_value_is_stored_as_the_declared_type_of_the_channel(written: object) -> None:
    """
    Arrange: the channels of a schema with a field whose type is a Pydantic model.
    Act: apply a write that Pydantic can parse into the model.
    Assert: the channel holds an instance of the model.
    """
    channels = build_channels(TotalState)

    actual = apply_updates(channels, {}, [{"total": written}])["total"]

    assert actual == Total(count=1), (
        f"Expected Total(count=1), got {actual!r}. The channel must hold its declared type."
    )


@pytest.mark.parametrize(
    ("written", "expectation"),
    [(1, does_not_raise()), (0, pytest.raises(ValueError)), (-1, pytest.raises(ValueError))],
    ids=["value-within-the-constraint", "value-at-the-limit", "value-outside-the-constraint"],
)
def test_constraint_in_the_annotation_of_a_field_applies_to_its_channel(
    build_schema: Callable[[object], object], written: int, expectation: AbstractContextManager[object]
) -> None:
    """
    Arrange: a schema of each kind with a field that has a Pydantic constraint in its annotation.
    Act: apply a write to the channel of the field.
    Assert: a ValueError is raised only when the value does not satisfy the constraint.
    """
    channels = build_channels(build_schema(Annotated[int, Field(gt=0)]))

    with expectation:
        apply_updates(channels, {}, [{"items": written}])


@pytest.mark.parametrize(
    "schema", [ConnectionDict, ConnectionDataclass, ConnectionModel], ids=["typed-dict", "dataclass", "model"]
)
def test_field_of_a_type_that_pydantic_cannot_validate_is_rejected(schema: object) -> None:
    """
    Arrange: a schema of each kind with a field whose type is a class for which Pydantic cannot make a validator.
    Act: build the channels of the schema.
    Assert: the PydanticSchemaGenerationError of Pydantic is raised.
    """
    with pytest.raises(PydanticSchemaGenerationError):
        build_channels(schema)

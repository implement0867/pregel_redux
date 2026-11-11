import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Annotated,
    NotRequired,
    Protocol,
    Required,
    Self,
    TypeVar,
    assert_never,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
    overload,
    runtime_checkable,
)

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from pydantic_core import to_json
from typing_extensions import TypeForm


class Missing:
    pass


MISSING = Missing()


@dataclass(init=False)
class Reducer[ValueT, UpdateT]:
    function: Callable[..., object]
    initial: object
    initial_factory: Callable[[], object] | None
    update_type: TypeForm[object] | Missing

    @overload
    def __init__(self, function: Callable[[ValueT, UpdateT], ValueT], *, initial: ValueT) -> None: ...
    @overload
    def __init__(
        self, function: Callable[[ValueT, UpdateT], ValueT], *, initial_factory: Callable[[], ValueT]
    ) -> None: ...
    @overload
    def __init__(
        self, function: Callable[[ValueT, UpdateT], ValueT], *, initial: ValueT, update_type: TypeForm[UpdateT]
    ) -> None: ...
    @overload
    def __init__(
        self,
        function: Callable[[ValueT, UpdateT], ValueT],
        *,
        initial_factory: Callable[[], ValueT],
        update_type: TypeForm[UpdateT],
    ) -> None: ...
    def __init__(
        self,
        function: Callable[..., object],
        *,
        initial: object = MISSING,
        initial_factory: Callable[[], object] | None = None,
        update_type: TypeForm[object] | Missing = MISSING,
    ) -> None:
        if (initial is MISSING) == (initial_factory is None):
            raise TypeError("a Reducer needs exactly one of initial and initial_factory")
        self.function = function
        self.initial = initial
        self.initial_factory = initial_factory
        self.update_type = update_type

    def make_initial(self) -> object:
        return deepcopy(self.initial) if self.initial_factory is None else self.initial_factory()


def dump_json(adapter: TypeAdapter[object], value: object) -> JsonValue:
    return adapter.dump_python(value, mode="json", by_alias=False, round_trip=True)


def load_json(adapter: TypeAdapter[object], value: JsonValue) -> object:
    return adapter.validate_json(to_json(value, inf_nan_mode="strings"), by_alias=False, by_name=True)


@dataclass(frozen=True)
class Stored[ValueT]:
    value: ValueT


@dataclass(frozen=True)
class Channel:
    name: str
    value_type: object
    reducer: Reducer[object, object] | None
    adapter: TypeAdapter[object] = field(compare=False, repr=False)
    update_adapter: TypeAdapter[object] = field(compare=False, repr=False)

    def parse(self, value: object) -> object:
        try:
            return self.adapter.validate_python(value)
        except ValidationError as error:
            raise ValueError(f"channel {self.name!r} cannot hold {value!r}, its type is {self.value_type!r}") from error

    def parse_update(self, value: object) -> object:
        try:
            return self.update_adapter.validate_python(value)
        except ValidationError as error:
            raise ValueError(f"channel {self.name!r} cannot take the update {value!r}") from error

    def open(self) -> "LastValue | BinaryOperatorAggregate":
        match self.reducer:
            case None:
                return LastValue(self, None)
            case Reducer():
                return BinaryOperatorAggregate(self, self.reducer, self.parse(self.reducer.make_initial()))
            case _:
                assert_never(self.reducer)


@dataclass(frozen=True)
class LastValue:
    spec: Channel
    stored: Stored[object] | None

    def get(self) -> Stored[object] | None:
        return self.stored

    def replace(self, value: object) -> "LastValue":
        return LastValue(self.spec, Stored(self.spec.parse(value)))

    def update(self, values: Sequence[object]) -> "LastValue | None":
        match values:
            case []:
                return None
            case [value]:
                return LastValue(self.spec, Stored(value))
            case _:
                raise ValueError(
                    f"channel {self.spec.name!r} received more than one value in one step and it has no reducer"
                )

    def checkpoint(self) -> Stored[JsonValue] | None:
        return None if self.stored is None else Stored(dump_json(self.spec.adapter, self.stored.value))

    def from_checkpoint(self, saved: JsonValue) -> "LastValue":
        return LastValue(self.spec, Stored(load_json(self.spec.adapter, saved)))


@dataclass(frozen=True)
class BinaryOperatorAggregate:
    spec: Channel
    reducer: Reducer[object, object]
    value: object

    def get(self) -> Stored[object] | None:
        return Stored(self.value)

    def replace(self, value: object) -> "BinaryOperatorAggregate":
        return BinaryOperatorAggregate(self.spec, self.reducer, self.spec.parse(value))

    def update(self, values: Sequence[object]) -> "BinaryOperatorAggregate | None":
        if not values:
            return None
        current = self.value
        for value in values:
            current = self.spec.parse(self.reducer.function(current, value))
        return BinaryOperatorAggregate(self.spec, self.reducer, current)

    def checkpoint(self) -> Stored[JsonValue] | None:
        return Stored(dump_json(self.spec.adapter, self.value))

    def from_checkpoint(self, saved: JsonValue) -> "BinaryOperatorAggregate":
        return BinaryOperatorAggregate(self.spec, self.reducer, load_json(self.spec.adapter, saved))


type FieldChannel = LastValue | BinaryOperatorAggregate


@runtime_checkable
class SubscriptableHint(Protocol):
    def __getitem__(self, arguments: tuple[object, ...]) -> Self: ...


def substitute_type_arguments[HintT](hint: HintT, arguments: Mapping[object, HintT]) -> HintT:
    match hint:
        case TypeVar():
            return arguments.get(hint, hint)
        case SubscriptableHint():
            parameters = tuple[object, ...](getattr(hint, "__parameters__", ()))
            if not parameters:
                return hint
            return hint[tuple(arguments.get(parameter, parameter) for parameter in parameters)]
        case _:
            return hint


def find_reducer(name: str, metadata: tuple[object, ...]) -> Reducer[object, object] | None:
    markers: list[Reducer[object, object]] = [item for item in metadata if isinstance(item, Reducer)]
    match markers:
        case []:
            return None
        case [marker]:
            return marker
        case _:
            raise TypeError(f"field {name!r} has more than one Reducer annotation")


def make_channel(
    name: str,
    value_type: object,
    annotation: TypeForm[object],
    metadata: tuple[object, ...],
    arguments: Mapping[object, TypeForm[object]],
) -> Channel:
    reducer = find_reducer(name, metadata)
    adapter = TypeAdapter[object](annotation)
    if reducer is None or isinstance(reducer.update_type, Missing):
        return Channel(name, value_type, reducer, adapter, adapter)
    return Channel(
        name, value_type, reducer, adapter, TypeAdapter(substitute_type_arguments(reducer.update_type, arguments))
    )


def build_channel(name: str, hint: TypeForm[object], arguments: Mapping[object, TypeForm[object]]) -> Channel:
    unwrapped = get_args(hint)[0] if get_origin(hint) in (Required, NotRequired) else hint
    if get_origin(unwrapped) is Annotated:
        value_type, *metadata = get_args(unwrapped)
        return make_channel(name, value_type, unwrapped, tuple(metadata), arguments)
    return make_channel(name, unwrapped, unwrapped, (), arguments)


def read_field_hints(schema_class: object) -> dict[str, TypeForm[object]]:
    if is_typeddict(schema_class):
        return get_type_hints(schema_class, include_extras=True)
    if dataclasses.is_dataclass(schema_class):
        hints = get_type_hints(schema_class, include_extras=True)
        return {field.name: hints[field.name] for field in dataclasses.fields(schema_class)}
    raise TypeError(f"{schema_class!r} is not a TypedDict, a dataclass or a Pydantic model")


def build_channels(schema: object) -> dict[str, Channel]:
    match schema:
        case type() if issubclass(schema, BaseModel):
            generic = schema.__pydantic_generic_metadata__
            origin = generic["origin"]
            parameters = () if origin is None else origin.__pydantic_generic_metadata__["parameters"]
            model_arguments: dict[object, TypeForm[object]] = dict(zip(parameters, generic["args"], strict=True))
            return {
                name: make_channel(
                    name, info.annotation, info.rebuild_annotation(), tuple(info.metadata), model_arguments
                )
                for name, info in schema.model_fields.items()
            }
        case _:
            origin = get_origin(schema)
            hints = read_field_hints(schema if origin is None else origin)
            arguments: dict[object, TypeForm[object]] = (
                {} if origin is None else dict(zip(origin.__parameters__, get_args(schema), strict=True))
            )
            return {
                name: build_channel(name, substitute_type_arguments(hint, arguments), arguments)
                for name, hint in hints.items()
            }


def merge_channels(schemas: Iterable[object]) -> dict[str, Channel]:
    merged: dict[str, Channel] = {}
    for schema in schemas:
        for name, channel in build_channels(schema).items():
            existing = merged.get(name)
            if existing is None or existing.reducer is None:
                merged[name] = channel
            elif channel.reducer is not None and channel.reducer != existing.reducer:
                raise ValueError(f"channel {name!r} is declared with two different reducers")
    return merged


def extract_writes(update: object) -> dict[str, object]:
    match update:
        case None:
            return {}
        case Mapping():
            return {**update}
        case BaseModel():
            return {name: getattr(update, name) for name in update.model_fields_set}
        case _ if dataclasses.is_dataclass(update) and not isinstance(update, type):
            defaults = {
                field.name: field.default if field.default_factory is dataclasses.MISSING else field.default_factory()
                for field in dataclasses.fields(update)
            }
            return {name: value for name, default in defaults.items() if (value := getattr(update, name)) != default}
        case _:
            raise TypeError(f"{update!r} is not a mapping, a dataclass instance, a Pydantic model or None")


def reject_unknown_channels(channels: Mapping[str, object], names: Iterable[str]) -> None:
    unknown = sorted(set(names) - channels.keys())
    if unknown:
        raise ValueError(f"{unknown!r} are not channels of this graph")


def parse_updates(channels: Mapping[str, Channel], updates: Mapping[str, object]) -> dict[str, object]:
    reject_unknown_channels(channels, updates)
    return {name: channels[name].parse_update(value) for name, value in updates.items()}

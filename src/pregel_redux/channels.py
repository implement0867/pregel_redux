import dataclasses
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import (
    Annotated,
    Literal,
    NotRequired,
    Protocol,
    Required,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
    overload,
    runtime_checkable,
)

from pydantic import BaseModel


class Unset(Enum):
    unset = "unset"


@dataclass(init=False)
class Reducer[ValueT, UpdateT]:
    function: Callable[..., object]
    initial: object
    initial_factory: Callable[[], object] | None

    @overload
    def __init__(self, function: Callable[[ValueT, UpdateT], ValueT], *, initial: ValueT) -> None: ...
    @overload
    def __init__(
        self, function: Callable[[ValueT, UpdateT], ValueT], *, initial_factory: Callable[[], ValueT]
    ) -> None: ...
    def __init__(
        self,
        function: Callable[[ValueT, UpdateT], ValueT],
        *,
        initial: ValueT | Literal[Unset.unset] = Unset.unset,
        initial_factory: Callable[[], ValueT] | None = None,
    ) -> None:
        if (initial is Unset.unset) == (initial_factory is None):
            raise TypeError("a Reducer needs exactly one of initial and initial_factory")
        self.function = function
        self.initial = initial
        self.initial_factory = initial_factory

    def make_initial(self) -> object:
        return deepcopy(self.initial) if self.initial_factory is None else self.initial_factory()


@dataclass(frozen=True)
class Channel:
    name: str
    value_type: object
    reducer: Reducer[object, object] | None


@runtime_checkable
class SubscriptableHint(Protocol):
    def __getitem__(self, arguments: tuple[object, ...]) -> object: ...


def substitute_type_arguments(hint: object, arguments: Mapping[object, object]) -> object:
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


def build_channel(name: str, hint: object) -> Channel:
    unwrapped = get_args(hint)[0] if get_origin(hint) in (Required, NotRequired) else hint
    if get_origin(unwrapped) is Annotated:
        value_type, *metadata = get_args(unwrapped)
        return Channel(name, value_type, find_reducer(name, tuple(metadata)))
    return Channel(name, unwrapped, None)


def read_field_hints(schema_class: object) -> dict[str, object]:
    if is_typeddict(schema_class):
        return get_type_hints(schema_class, include_extras=True)
    if dataclasses.is_dataclass(schema_class):
        hints = get_type_hints(schema_class, include_extras=True)
        return {field.name: hints[field.name] for field in dataclasses.fields(schema_class)}
    raise TypeError(f"{schema_class!r} is not a TypedDict, a dataclass or a Pydantic model")


def build_channels(schema: object) -> dict[str, Channel]:
    match schema:
        case type() if issubclass(schema, BaseModel):
            return {
                name: Channel(name, field.annotation, find_reducer(name, tuple(field.metadata)))
                for name, field in schema.model_fields.items()
            }
        case _:
            origin = get_origin(schema)
            hints = read_field_hints(schema if origin is None else origin)
            arguments: dict[object, object] = (
                {} if origin is None else dict(zip(origin.__parameters__, get_args(schema), strict=True))
            )
            return {
                name: build_channel(name, substitute_type_arguments(hint, arguments)) for name, hint in hints.items()
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


def reject_unknown_channels(channels: Mapping[str, Channel], names: Iterable[str]) -> None:
    unknown = sorted(set(names) - channels.keys())
    if unknown:
        raise ValueError(f"{unknown!r} are not channels of this graph")


def apply_writes(
    channels: Mapping[str, Channel], values: Mapping[str, object], updates: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    result = dict(values)
    written: set[str] = set()
    for update in updates:
        reject_unknown_channels(channels, update)
        for name, value in update.items():
            reducer = channels[name].reducer
            if reducer is not None:
                result[name] = reducer.function(result[name], value)
            elif name in written:
                raise ValueError(f"channel {name!r} received more than one value in one step and it has no reducer")
            else:
                result[name] = value
            written.add(name)
    return result

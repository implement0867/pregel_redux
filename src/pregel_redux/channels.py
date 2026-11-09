import dataclasses
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, Required, TypeVar, get_args, get_origin, get_type_hints, is_typeddict

from pydantic import BaseModel


@dataclass(frozen=True)
class Channel:
    name: str
    value_type: Any
    reducer: Callable[[Any, Any], Any] | None


def substitute_type_arguments(hint: Any, arguments: Mapping[Any, Any]) -> Any:
    match hint:
        case TypeVar():
            return arguments.get(hint, hint)
        case _:
            parameters = getattr(hint, "__parameters__", ())
            if not parameters:
                return hint
            return hint[tuple(arguments.get(parameter, parameter) for parameter in parameters)]


def reducer_from_metadata(metadata: tuple[Any, ...]) -> Callable[[Any, Any], Any] | None:
    if not metadata or not callable(metadata[-1]):
        return None
    positional = [
        parameter
        for parameter in inspect.signature(metadata[-1]).parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return metadata[-1] if len(positional) == 2 else None


def channel_from_hint(name: str, hint: Any) -> Channel:
    unwrapped = get_args(hint)[0] if get_origin(hint) in (Required, NotRequired) else hint
    if get_origin(unwrapped) is Annotated:
        return Channel(name, unwrapped.__origin__, reducer_from_metadata(unwrapped.__metadata__))
    return Channel(name, unwrapped, None)


def field_hints(schema_class: object) -> dict[str, Any]:
    if is_typeddict(schema_class):
        return get_type_hints(schema_class, include_extras=True)
    if dataclasses.is_dataclass(schema_class):
        hints = get_type_hints(schema_class, include_extras=True)
        return {field.name: hints[field.name] for field in dataclasses.fields(schema_class)}
    raise TypeError(f"{schema_class!r} is not a TypedDict, a dataclass or a Pydantic model")


def channels_from_schema(schema: object) -> dict[str, Channel]:
    match schema:
        case type() if issubclass(schema, BaseModel):
            return {
                name: Channel(name, field.annotation, reducer_from_metadata(tuple(field.metadata)))
                for name, field in schema.model_fields.items()
            }
        case _:
            origin = get_origin(schema)
            hints = field_hints(schema if origin is None else origin)
            arguments: dict[Any, Any] = (
                {} if origin is None else dict(zip(origin.__parameters__, get_args(schema), strict=True))
            )
            return {
                name: channel_from_hint(name, substitute_type_arguments(hint, arguments))
                for name, hint in hints.items()
            }


def merge_channels(schemas: Iterable[Any]) -> dict[str, Channel]:
    merged: dict[str, Channel] = {}
    for schema in schemas:
        for name, channel in channels_from_schema(schema).items():
            existing = merged.get(name)
            if existing is None or existing.reducer is None:
                merged[name] = channel
            elif channel.reducer is not None and channel.reducer != existing.reducer:
                raise ValueError(f"channel {name!r} is declared with two different reducers")
    return merged


def writes_from_update(update: Any) -> dict[str, Any]:
    match update:
        case None:
            return {}
        case Mapping():
            return {**update}
        case BaseModel():
            return {name: getattr(update, name) for name in update.model_fields_set}
        case _ if dataclasses.is_dataclass(update) and not isinstance(update, type):
            return {field.name: getattr(update, field.name) for field in dataclasses.fields(update)}
        case _:
            raise TypeError(f"{update!r} is not a mapping, a dataclass instance, a Pydantic model or None")


def apply_writes(
    channels: Mapping[str, Channel], values: Mapping[str, Any], updates: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    result = dict(values)
    written: set[str] = set()
    for update in updates:
        for name, value in update.items():
            if name not in channels:
                raise ValueError(f"{name!r} is not a channel of this graph")
            reducer = channels[name].reducer
            if reducer is None:
                if name in written:
                    raise ValueError(f"channel {name!r} received more than one value in one step and it has no reducer")
                result[name] = value
            elif name in result:
                result[name] = reducer(result[name], value)
            else:
                result[name] = value
            written.add(name)
    return result

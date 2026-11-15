import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, get_args, get_origin

from pregel_redux.channels import extract_writes


class Endpoint(Enum):
    """Replaces a node name in `StateGraph.add_edge` and in `Command.goto`."""

    start = "start"
    end = "end"


@dataclass(frozen=True)
class Send:
    """Runs `node` once with `arg` as its input in place of the graph state.

    The task runs in the step after the node that returns the `Send`.
    """

    node: str
    arg: object


type Target[NameT: str] = NameT | Literal[Endpoint.end] | Send


@dataclass(frozen=True, kw_only=True)
class Command[NameT: str]:
    """Writes `update` to the state and selects the nodes of the next step.

    A node declares its `goto` targets as `Literal` arguments of `Command` in its return annotation.
    `StateGraph.compile` checks each declared target against the nodes of the graph.

    Parameters
    ----------
    update
        A mapping, a dataclass instance or a Pydantic model.
        A dataclass instance writes each field that differs from its default.
        A Pydantic model writes each field that the caller set.
    goto
        The node names add to the static edges of the node.
        `Endpoint.end` leaves the targets unchanged.
    """

    update: object = None
    goto: Target[NameT] | Sequence[Target[NameT]] = ()


@dataclass(frozen=True)
class Routes:
    targets: frozenset[str]
    sends: tuple[Send, ...]


@dataclass(frozen=True)
class Outcome:
    writes: Mapping[str, object]
    routes: Routes


def resolve_goto(goto: object, node_names: Collection[str]) -> Routes:
    selections: Sequence[object]
    match goto:
        case str() | Endpoint() | Send():
            selections = [goto]
        case Sequence():
            selections = goto
        case _:
            raise TypeError(f"goto is {goto!r}, which is not a node name, Endpoint.end, a Send or a sequence of them")
    names: set[str] = set()
    sends: list[Send] = []
    for selection in selections:
        match selection:
            case Endpoint.end:
                continue
            case str():
                names.add(selection)
            case Send():
                sends.append(selection)
            case _:
                raise TypeError(f"goto contains {selection!r}, which is not a node name, Endpoint.end or a Send")
    unknown = sorted({*names, *(send.node for send in sends)} - set(node_names))
    if unknown:
        raise ValueError(f"goto refers to {unknown!r}, which are not nodes of this graph")
    return Routes(frozenset(names), tuple(sends))


def read_outcome(result: object, static_targets: frozenset[str], node_names: Collection[str]) -> Outcome:
    match result:
        case Command():
            routes = resolve_goto(result.goto, node_names)
            return Outcome(extract_writes(result.update), Routes(static_targets | routes.targets, routes.sends))
        case _:
            return Outcome(extract_writes(result), Routes(static_targets, ()))


def collect_literal_names(hint: object, *, inside_command: bool) -> frozenset[str]:
    origin = get_origin(hint)
    if origin is Literal:
        names = (argument for argument in get_args(hint) if isinstance(argument, str))
        return frozenset(names) if inside_command else frozenset()
    inside = inside_command or origin is Command
    return frozenset[str]().union(
        *(collect_literal_names(argument, inside_command=inside) for argument in get_args(hint))
    )


def read_declared_targets(function: Callable[..., object]) -> frozenset[str]:
    annotation = inspect.signature(function, eval_str=True).return_annotation
    return collect_literal_names(annotation, inside_command=False)

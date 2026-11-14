from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass

from pregel_redux.interrupts import Interrupt
from pregel_redux.snapshots import StateSnapshot


@dataclass(frozen=True)
class ValuesPart:
    values: Mapping[str, object]
    interrupts: tuple[Interrupt, ...]


@dataclass(frozen=True)
class UpdatesPart:
    node: str
    update: Mapping[str, object]


@dataclass(frozen=True)
class InterruptsPart:
    interrupts: tuple[Interrupt, ...]


@dataclass(frozen=True)
class CustomPart:
    data: object


@dataclass(frozen=True)
class CheckpointPart:
    state: StateSnapshot


@dataclass(frozen=True)
class TaskStartPart:
    id: str
    name: str
    input: object


@dataclass(frozen=True)
class TaskResultPart:
    id: str
    name: str
    update: Mapping[str, object]
    interrupts: tuple[Interrupt, ...]


type StreamPart = (
    ValuesPart | UpdatesPart | InterruptsPart | CustomPart | CheckpointPart | TaskStartPart | TaskResultPart
)


@dataclass(frozen=True)
class ChildPart:
    ns: tuple[str, *tuple[str, ...]]
    part: StreamPart


@dataclass(frozen=True)
class Stream:
    emit: Callable[[StreamPart | ChildPart], None]

    def send(self, ns: tuple[str, ...], part: StreamPart) -> None:
        match ns:
            case ():
                self.emit(part)
            case (first, *rest):
                self.emit(ChildPart((first, *rest), part))
            case _:
                raise TypeError(f"the namespace {ns!r} is not a tuple")


current_stream_writer = ContextVar[Callable[[object], None]]("current_stream_writer")


def get_stream_writer() -> Callable[[object], None]:
    try:
        return current_stream_writer.get()
    except LookupError:
        raise RuntimeError("get_stream_writer works only inside a node of a running graph") from None

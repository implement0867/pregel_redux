from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum

from pregel_redux.interrupts import Interrupt
from pregel_redux.snapshots import StateSnapshot


class StreamMode(StrEnum):
    values = "values"
    updates = "updates"
    custom = "custom"
    checkpoints = "checkpoints"
    tasks = "tasks"


@dataclass(frozen=True)
class ValuesPart:
    ns: tuple[str, ...]
    values: Mapping[str, object]
    interrupts: tuple[Interrupt, ...]


@dataclass(frozen=True)
class UpdatesPart:
    ns: tuple[str, ...]
    node: str
    update: Mapping[str, object]


@dataclass(frozen=True)
class InterruptsPart:
    ns: tuple[str, ...]
    interrupts: tuple[Interrupt, ...]


@dataclass(frozen=True)
class CustomPart:
    ns: tuple[str, ...]
    data: object


@dataclass(frozen=True)
class CheckpointPart:
    ns: tuple[str, ...]
    state: StateSnapshot


@dataclass(frozen=True)
class TaskStartPart:
    ns: tuple[str, ...]
    id: str
    name: str
    input: object


@dataclass(frozen=True)
class TaskResultPart:
    ns: tuple[str, ...]
    id: str
    name: str
    update: Mapping[str, object]
    interrupts: tuple[Interrupt, ...]


type StreamPart = (
    ValuesPart | UpdatesPart | InterruptsPart | CustomPart | CheckpointPart | TaskStartPart | TaskResultPart
)


@dataclass(frozen=True)
class Stream:
    modes: frozenset[StreamMode]
    subgraphs: bool
    emit: Callable[[StreamPart], None]

    def send(self, mode: StreamMode, part: StreamPart) -> None:
        if mode in self.modes and (self.subgraphs or not part.ns):
            self.emit(part)


current_stream_writer = ContextVar[Callable[[object], None]]("current_stream_writer")


def get_stream_writer() -> Callable[[object], None]:
    try:
        return current_stream_writer.get()
    except LookupError:
        raise RuntimeError("get_stream_writer works only inside a node of a running graph") from None

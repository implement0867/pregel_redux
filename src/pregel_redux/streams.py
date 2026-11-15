from collections.abc import Callable, Mapping
from dataclasses import dataclass

from typing_extensions import TypeForm

from pregel_redux.configs import Namespace, Segment
from pregel_redux.messages import Entrance
from pregel_redux.results import StopReason
from pregel_redux.snapshots import StateSnapshot


@dataclass(frozen=True)
class ValuesPart:
    values: Mapping[str, object]


@dataclass(frozen=True)
class UpdatesPart:
    node: str
    update: Mapping[str, object]


@dataclass(frozen=True)
class WaitingPart:
    accepts: Mapping[Entrance, TypeForm[object]]


@dataclass(frozen=True)
class StoppedPart:
    reason: StopReason


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


type StreamPart = (
    ValuesPart | UpdatesPart | WaitingPart | StoppedPart | CustomPart | CheckpointPart | TaskStartPart | TaskResultPart
)


@dataclass(frozen=True)
class ChildPart:
    ns: tuple[Segment, *tuple[Segment, ...]]
    part: StreamPart


@dataclass(frozen=True)
class Stream:
    emit: Callable[[StreamPart | ChildPart], None]

    def send(self, ns: Namespace, part: StreamPart) -> None:
        match ns:
            case ():
                self.emit(part)
            case (first, *rest):
                self.emit(ChildPart((first, *rest), part))
            case _:
                raise TypeError(f"the namespace {ns!r} is not a tuple")


@dataclass(frozen=True)
class StreamWriter:
    stream: Stream
    ns: Namespace

    def __call__(self, data: object) -> None:
        self.stream.send(self.ns, CustomPart(data))

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from typing_extensions import TypeForm

from pregel_redux.configs import Namespace, Segment
from pregel_redux.messages import Entrance
from pregel_redux.results import StopReason
from pregel_redux.snapshots import StateSnapshot


@dataclass(frozen=True)
class ValuesPart:
    """Holds each state value after a step."""

    values: Mapping[str, object]


@dataclass(frozen=True)
class UpdatesPart:
    """Holds the state writes of one node after the node returns."""

    node: str
    update: Mapping[str, object]


@dataclass(frozen=True)
class WaitingPart:
    """Marks the end of a run whose nodes wait for a message."""

    accepts: Mapping[Entrance, TypeForm[object]]


@dataclass(frozen=True)
class StoppedPart:
    """Marks the end of a run that stopped before its end."""

    reason: StopReason


@dataclass(frozen=True)
class CustomPart:
    """Holds the value that a node passes to its `StreamWriter`."""

    data: object


@dataclass(frozen=True)
class CheckpointPart:
    """Holds the snapshot of each saved checkpoint of a `PersistentGraph`."""

    state: StateSnapshot


@dataclass(frozen=True)
class TaskStartPart:
    """Holds the input of a node task before the task runs."""

    id: str
    name: str
    input: object


@dataclass(frozen=True)
class TaskResultPart:
    """Holds the state writes of a node task after the task returns."""

    id: str
    name: str
    update: Mapping[str, object]


type StreamPart = (
    ValuesPart | UpdatesPart | WaitingPart | StoppedPart | CustomPart | CheckpointPart | TaskStartPart | TaskResultPart
)


@dataclass(frozen=True)
class ChildPart:
    """Wraps a part that a child graph emits.

    `ns` identifies the child graph.
    """

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
    """Emits a `CustomPart` in the stream of the run.

    A node receives it through a parameter annotated `StreamWriter`.
    """

    stream: Stream
    ns: Namespace

    def __call__(self, data: object) -> None:
        self.stream.send(self.ns, CustomPart(data))

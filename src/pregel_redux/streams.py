from collections.abc import Callable, Mapping
from dataclasses import dataclass

from typing_extensions import TypeForm

from pregel_redux.configs import Namespace, Segment
from pregel_redux.messages import Entrance
from pregel_redux.results import StopReason
from pregel_redux.snapshots import StateSnapshot


@dataclass(frozen=True)
class ValuesPart:
    """Follows each step with each state value and precedes each `StoppedPart`."""

    values: Mapping[str, object]


@dataclass(frozen=True)
class UpdatesPart:
    """Follows each node that returns."""

    node: str
    update: Mapping[str, object]


@dataclass(frozen=True)
class WaitingPart:
    """Ends the stream of a run whose nodes wait for a message."""

    accepts: Mapping[Entrance, TypeForm[object]]


@dataclass(frozen=True)
class StoppedPart:
    """Ends the stream of a stopped run."""

    reason: StopReason


@dataclass(frozen=True)
class CustomPart:
    """Follows each call to a `StreamWriter`."""

    data: object


@dataclass(frozen=True)
class CheckpointPart:
    """Follows each checkpoint that a `PersistentGraph` saves."""

    state: StateSnapshot


@dataclass(frozen=True)
class TaskStartPart:
    """Precedes each node task."""

    id: str
    name: str
    input: object


@dataclass(frozen=True)
class TaskResultPart:
    """Follows each node task that returns."""

    id: str
    name: str
    update: Mapping[str, object]


type StreamPart = (
    ValuesPart | UpdatesPart | WaitingPart | StoppedPart | CustomPart | CheckpointPart | TaskStartPart | TaskResultPart
)


@dataclass(frozen=True)
class ChildPart:
    """Prefixes a part that a subgraph emits with the namespace `ns` of the subgraph."""

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

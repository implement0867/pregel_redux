from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from typing_extensions import TypeForm

from pregel_redux.messages import Entrance


@dataclass(frozen=True)
class Finished[OutputT]:
    """Holds the output of a run that reached its end.

    `output` is an instance of the output schema of the graph.
    """

    output: OutputT


@dataclass(frozen=True)
class Waiting:
    """Holds the state of a run whose nodes wait for a message.

    A `Message` continues the run.

    Attributes
    ----------
    accepts
        Maps each waiting node to the type of the message that it receives.
    """

    values: Mapping[str, object]
    accepts: Mapping[Entrance, TypeForm[object]]


class StopReason(Enum):
    """Names the cause of a `Stopped` run.

    `breakpoint` marks a node in `interrupt_before` or `interrupt_after`.
    `shutdown` marks a shutdown request from the shutdown signal of the run.
    """

    breakpoint = "breakpoint"
    shutdown = "shutdown"


@dataclass(frozen=True)
class Stopped:
    """Holds the state of a run that stopped before its end.

    A later run on the same thread with the input `None` continues from the stop.
    """

    values: Mapping[str, object]
    reason: StopReason

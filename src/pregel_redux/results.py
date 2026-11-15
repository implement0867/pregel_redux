from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from typing_extensions import TypeForm

from pregel_redux.messages import Entrance


@dataclass(frozen=True)
class Finished[OutputT]:
    """`ainvoke` returns `Finished` for a run that reached its end.

    `output` is an instance of the output schema of the graph.
    """

    output: OutputT


@dataclass(frozen=True)
class Waiting:
    """`ainvoke` returns `Waiting` for a run whose nodes wait for a message.

    A `Message` continues the run.
    `accepts` maps each waiting node to its message type.
    """

    values: Mapping[str, object]
    accepts: Mapping[Entrance, TypeForm[object]]


class StopReason(Enum):
    """Classifies the cause of a `Stopped` run.

    A node in `interrupt_before` or `interrupt_after` causes a `breakpoint` stop.
    A shutdown request from the shutdown signal of the run causes a `shutdown` stop.
    """

    breakpoint = "breakpoint"
    shutdown = "shutdown"


@dataclass(frozen=True)
class Stopped:
    """`ainvoke` returns `Stopped` for a run that stopped before its end.

    A later run on the same thread with the input `None` continues from the stop.
    """

    values: Mapping[str, object]
    reason: StopReason

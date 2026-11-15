from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from typing_extensions import TypeForm

from pregel_redux.messages import Entrance


@dataclass(frozen=True)
class Finished[OutputT]:
    output: OutputT


@dataclass(frozen=True)
class Waiting:
    values: Mapping[str, object]
    accepts: Mapping[Entrance, TypeForm[object]]


class StopReason(Enum):
    breakpoint = "breakpoint"
    shutdown = "shutdown"


@dataclass(frozen=True)
class Stopped:
    values: Mapping[str, object]
    reason: StopReason

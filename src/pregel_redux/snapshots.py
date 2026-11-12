from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from pregel_redux.checkpoints import CheckpointSource
from pregel_redux.interrupts import Interrupt


@dataclass(frozen=True)
class TaskSnapshot:
    id: str
    name: str
    interrupts: tuple[Interrupt, ...]
    finished: bool
    state: "StateSnapshot | None"


@dataclass(frozen=True)
class StateSnapshot:
    values: Mapping[str, object]
    next: tuple[str, ...]
    checkpoint_ns: str
    checkpoint_id: UUID
    parent_id: UUID | None
    created_at: datetime
    step: int
    source: CheckpointSource
    tasks: tuple[TaskSnapshot, ...]
    interrupts: tuple[Interrupt, ...]

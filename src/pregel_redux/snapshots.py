from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import assert_never
from uuid import UUID

from pregel_redux.checkpoints import CheckpointSource
from pregel_redux.interrupts import Interrupt
from pregel_redux.task_refs import TaskRef


@dataclass(frozen=True)
class PendingTask:
    pass


@dataclass(frozen=True)
class FinishedTask:
    result: Mapping[str, object]


@dataclass(frozen=True)
class WaitingTask:
    interrupts: tuple[Interrupt, *tuple[Interrupt, ...]]


@dataclass(frozen=True)
class FailedTask:
    error: str
    interrupts: tuple[Interrupt, ...]


type TaskStatus = PendingTask | FinishedTask | WaitingTask | FailedTask


@dataclass(frozen=True)
class TaskSnapshot:
    id: str
    path: TaskRef
    status: TaskStatus
    state: "StateSnapshot | None"

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def interrupts(self) -> tuple[Interrupt, ...]:
        match self.status:
            case PendingTask() | FinishedTask():
                return ()
            case WaitingTask() | FailedTask():
                return self.status.interrupts
            case _:
                assert_never(self.status)


@dataclass(frozen=True)
class StateSnapshot:
    values: Mapping[str, object]
    checkpoint_ns: str
    checkpoint_id: UUID
    parent_id: UUID | None
    created_at: datetime
    step: int
    source: CheckpointSource
    tasks: tuple[TaskSnapshot, ...]

    @property
    def next(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.tasks)

    @property
    def interrupts(self) -> tuple[Interrupt, ...]:
        return tuple(item for task in self.tasks for item in task.interrupts)

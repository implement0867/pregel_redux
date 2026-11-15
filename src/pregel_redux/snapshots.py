from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from typing_extensions import TypeForm

from pregel_redux.checkpoints import CheckpointSource
from pregel_redux.configs import Namespace
from pregel_redux.messages import Entrance
from pregel_redux.task_refs import TaskRef


@dataclass(frozen=True)
class PendingTask:
    pass


@dataclass(frozen=True)
class FinishedTask:
    result: Mapping[str, object]


@dataclass(frozen=True)
class WaitingTask:
    accepts: Mapping[Entrance, TypeForm[object]]


@dataclass(frozen=True)
class FailedTask:
    error: str
    accepts: Mapping[Entrance, TypeForm[object]]


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


@dataclass(frozen=True)
class StateSnapshot:
    values: Mapping[str, object]
    checkpoint_ns: Namespace
    checkpoint_id: UUID
    parent_id: UUID | None
    created_at: datetime
    step: int
    source: CheckpointSource
    tasks: tuple[TaskSnapshot, ...]
    accepts: Mapping[Entrance, TypeForm[object]]

    @property
    def next(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.tasks)

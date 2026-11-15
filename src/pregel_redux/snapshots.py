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
    """Marks a task that waits to run."""


@dataclass(frozen=True)
class FinishedTask:
    """Holds the state writes of a task that returned."""

    result: Mapping[str, object]


@dataclass(frozen=True)
class WaitingTask:
    """Marks a task whose child graph waits for a message.

    `accepts` maps each waiting node of the child graph to its message type.
    """

    accepts: Mapping[Entrance, TypeForm[object]]


@dataclass(frozen=True)
class FailedTask:
    """Holds the error of a task that raised an exception.

    `accepts` maps each waiting node of the child graph of the task to its message type.
    """

    error: str
    accepts: Mapping[Entrance, TypeForm[object]]


type TaskStatus = PendingTask | FinishedTask | WaitingTask | FailedTask


@dataclass(frozen=True)
class TaskSnapshot:
    """Describes one task of the next step.

    Attributes
    ----------
    state
        The snapshot of the child graph that the task runs.
        `state` is `None` for a node function and for a snapshot read with `subgraphs` set to `False`.
    """

    id: str
    path: TaskRef
    status: TaskStatus
    state: "StateSnapshot | None"

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class StateSnapshot:
    """Describes the state of one graph at one checkpoint.

    Attributes
    ----------
    tasks
        The tasks of the next step.
    accepts
        Maps each waiting node to the type of the message that it receives.
    """

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

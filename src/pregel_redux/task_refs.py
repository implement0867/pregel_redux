from dataclasses import dataclass
from typing import Literal, assert_never

from pregel_redux.channels import Stored
from pregel_redux.pregel import PullTask, PushTask, StartTask, Task


@dataclass(frozen=True)
class StartRef:
    """Identifies the task that writes the run input."""

    name: Literal["__start__"] = "__start__"


@dataclass(frozen=True)
class PullRef:
    """Identifies the task that runs a node after its edges trigger."""

    name: str


@dataclass(frozen=True)
class PushRef:
    """Identifies the task that runs a node for a `Send`.

    `index` is the position of the `Send` among the sends of the step.
    """

    name: str
    index: int


type TaskRef = StartRef | PullRef | PushRef


@dataclass(frozen=True)
class NodeCall:
    task: TaskRef
    node_input: object
    received: Stored[object] | None


def name_task(task: TaskRef) -> str:
    match task:
        case StartRef():
            return task.name
        case PullRef():
            return f"pull|{task.name}"
        case PushRef():
            return f"push|{task.name}|{task.index}"
        case _:
            assert_never(task)


def refer_task(task: Task) -> TaskRef:
    match task:
        case StartTask():
            return StartRef()
        case PullTask():
            return PullRef(task.node)
        case PushTask():
            return PushRef(task.send.node, task.index)
        case _:
            assert_never(task)

from dataclasses import dataclass
from typing import Literal, assert_never

from pregel_redux.pregel import PullTask, PushTask, StartTask, Task


@dataclass(frozen=True)
class StartRef:
    name: Literal["__start__"] = "__start__"


@dataclass(frozen=True)
class PullRef:
    name: str


@dataclass(frozen=True)
class PushRef:
    name: str
    index: int


type TaskRef = StartRef | PullRef | PushRef


@dataclass(frozen=True)
class NodeCall:
    task: TaskRef
    node_input: object


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

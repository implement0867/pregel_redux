from dataclasses import dataclass
from typing import assert_never

from pregel_redux.commands import Endpoint
from pregel_redux.pregel import NodeName, PullTask, PushTask, StartTask, Task


@dataclass(frozen=True)
class TaskRef:
    node: NodeName
    send_index: int | None


@dataclass(frozen=True)
class NodeCall:
    task: TaskRef
    node_input: object


def name_task(task: TaskRef) -> str:
    match task.node:
        case Endpoint.start:
            return "start"
        case str() as node:
            return f"pull|{node}" if task.send_index is None else f"push|{node}|{task.send_index}"
        case _:
            assert_never(task.node)


def refer_task(task: Task) -> TaskRef:
    match task:
        case StartTask():
            return TaskRef(Endpoint.start, None)
        case PullTask():
            return TaskRef(task.node, None)
        case PushTask():
            return TaskRef(task.send.node, task.index)
        case _:
            assert_never(task)

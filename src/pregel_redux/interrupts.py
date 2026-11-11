from contextvars import ContextVar
from dataclasses import dataclass
from typing import assert_never

from pydantic import JsonValue, TypeAdapter
from typing_extensions import TypeForm

from pregel_redux.checkpoints import Record, TaskAddress


class InterruptId(Record):
    task: TaskAddress
    index: int


class Interrupt(Record):
    interrupt_id: InterruptId
    node: str
    payload: JsonValue


class Answer(Record):
    interrupt_id: InterruptId
    value: JsonValue


class NodeInterrupt(BaseException):
    def __init__(self, payload: JsonValue) -> None:
        super().__init__(payload)
        self.payload = payload


@dataclass
class TaskInterrupts:
    answers: tuple[str, ...]
    asked: int = 0


current_interrupts = ContextVar[TaskInterrupts]("current_interrupts")


def interrupt[AnswerT](payload: JsonValue, answer_type: TypeForm[AnswerT]) -> AnswerT:
    try:
        task = current_interrupts.get()
    except LookupError:
        raise RuntimeError("interrupt works only inside a node of a running graph") from None
    index = task.asked
    task.asked += 1
    if index >= len(task.answers):
        raise NodeInterrupt(payload)
    return TypeAdapter[AnswerT](answer_type).validate_json(task.answers[index])


def find_interruption(group: BaseExceptionGroup) -> NodeInterrupt:
    interruptions, rest = group.split(NodeInterrupt)
    if interruptions is None or rest is not None:
        raise group
    return first_interruption(interruptions)


def first_interruption(group: BaseExceptionGroup[NodeInterrupt]) -> NodeInterrupt:
    first = group.exceptions[0]
    match first:
        case NodeInterrupt():
            return first
        case BaseExceptionGroup():
            return first_interruption(first)
        case _:
            assert_never(first)

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from uuid import NAMESPACE_OID, uuid5

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from typing_extensions import TypeForm

from pregel_redux.channels import Stored


class Interrupt(BaseModel):
    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    value: JsonValue
    id: str


class GraphInterrupt(Exception):
    def __init__(self, interrupts: tuple[Interrupt, ...]) -> None:
        super().__init__(interrupts)
        self.interrupts = interrupts


@dataclass(frozen=True)
class Finished[OutputT]:
    output: OutputT


class StopReason(Enum):
    question = "question"
    breakpoint = "breakpoint"
    shutdown = "shutdown"


@dataclass(frozen=True)
class Interrupted:
    interrupts: tuple[Interrupt, ...]
    values: Mapping[str, object]
    reason: StopReason


def make_interrupt_id(task_ns: str, index: int) -> str:
    return str(uuid5(NAMESPACE_OID, f"{task_ns}|{index}"))


@dataclass
class ResumeInput:
    values: list[JsonValue]
    by_interrupt_id: Mapping[str, JsonValue]


@dataclass
class Scratchpad:
    task_ns: str
    resume: list[JsonValue]
    run_resume: ResumeInput
    interrupt_counter: int

    def take_resume[ResponseT](self, index: int, parse: Callable[[JsonValue], ResponseT]) -> Stored[ResponseT] | None:
        interrupt_id = make_interrupt_id(self.task_ns, index)
        if interrupt_id in self.run_resume.by_interrupt_id and index <= len(self.resume):
            answer = Stored(parse(self.run_resume.by_interrupt_id[interrupt_id]))
            self.resume[index:] = [self.run_resume.by_interrupt_id[interrupt_id]]
            return answer
        if index < len(self.resume):
            return Stored(parse(self.resume[index]))
        if self.run_resume.values:
            answer = Stored(parse(self.run_resume.values[-1]))
            self.resume.append(self.run_resume.values.pop())
            return answer
        return None


current_scratchpad = ContextVar[Scratchpad]("current_scratchpad")


def interrupt[ResponseT](value: JsonValue, *, response_schema: TypeForm[ResponseT]) -> ResponseT:
    try:
        scratchpad = current_scratchpad.get()
    except LookupError:
        raise RuntimeError("interrupt works only inside a node of a running graph") from None
    index = scratchpad.interrupt_counter
    scratchpad.interrupt_counter += 1
    answer = scratchpad.take_resume(index, TypeAdapter[ResponseT](response_schema).validate_python)
    if answer is not None:
        return answer.value
    raise GraphInterrupt((Interrupt(value=value, id=make_interrupt_id(scratchpad.task_ns, index)),))

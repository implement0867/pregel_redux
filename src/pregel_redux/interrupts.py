from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import NAMESPACE_OID, uuid5

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from typing_extensions import TypeForm


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


@dataclass(frozen=True)
class Interrupted:
    interrupts: tuple[Interrupt, ...]
    values: Mapping[str, object]


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

    def make_interrupt_id(self, index: int) -> str:
        return str(uuid5(NAMESPACE_OID, f"{self.task_ns}|{index}"))

    def find_resume(self, index: int) -> bool:
        if index < len(self.resume):
            return True
        interrupt_id = self.make_interrupt_id(index)
        if interrupt_id in self.run_resume.by_interrupt_id:
            self.resume.append(self.run_resume.by_interrupt_id[interrupt_id])
            return True
        if self.run_resume.values:
            self.resume.append(self.run_resume.values.pop())
            return True
        return False


current_scratchpad = ContextVar[Scratchpad]("current_scratchpad")


def interrupt[ResponseT](value: JsonValue, *, response_schema: TypeForm[ResponseT]) -> ResponseT:
    try:
        scratchpad = current_scratchpad.get()
    except LookupError:
        raise RuntimeError("interrupt works only inside a node of a running graph") from None
    index = scratchpad.interrupt_counter
    scratchpad.interrupt_counter += 1
    if scratchpad.find_resume(index):
        return TypeAdapter[ResponseT](response_schema).validate_python(scratchpad.resume[index])
    raise GraphInterrupt((Interrupt(value=value, id=scratchpad.make_interrupt_id(index)),))

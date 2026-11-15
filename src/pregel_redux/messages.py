from dataclasses import dataclass, field

from pydantic import TypeAdapter
from typing_extensions import TypeForm

from pregel_redux.configs import Namespace


@dataclass(frozen=True)
class Entrance:
    checkpoint_ns: Namespace
    node: str


@dataclass(frozen=True)
class Message:
    value: object
    to: Entrance | None = None


@dataclass(frozen=True)
class MessageSchema:
    message_type: TypeForm[object]
    adapter: TypeAdapter[object] = field(compare=False, repr=False)

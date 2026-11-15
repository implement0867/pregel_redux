from dataclasses import dataclass, field

from pydantic import TypeAdapter
from typing_extensions import TypeForm

from pregel_redux.configs import Namespace


@dataclass(frozen=True)
class Entrance:
    """Identifies a node that waits for a message and the graph that holds the node."""

    checkpoint_ns: Namespace
    node: str


@dataclass(frozen=True)
class Message:
    """Delivers `value` to a node that waits for a message.

    A node waits for a message through a parameter annotated `Received[T]`.
    The node validates `value` against `T`.

    Parameters
    ----------
    to
        The node that receives `value`.
        `None` selects the only waiting node of the thread.
    """

    value: object
    to: Entrance | None = None


@dataclass(frozen=True)
class MessageSchema:
    message_type: TypeForm[object]
    adapter: TypeAdapter[object] = field(compare=False, repr=False)

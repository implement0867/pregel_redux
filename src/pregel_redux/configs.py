from typing import NotRequired, TypedDict
from uuid import UUID


class ThreadConfig(TypedDict):
    thread_id: str
    checkpoint_id: NotRequired[UUID]
    checkpoint_ns: NotRequired[str]


def read_config() -> ThreadConfig:
    raise RuntimeError("read_config supplies a value only in a run of a graph that has a checkpointer")

from dataclasses import dataclass
from typing import NotRequired, TypedDict, assert_never
from uuid import UUID


class ThreadConfig(TypedDict):
    thread_id: str
    checkpoint_id: NotRequired[UUID | str]
    checkpoint_ns: NotRequired[str]


@dataclass(frozen=True)
class RunTarget:
    thread_id: str
    fork_from: UUID | None


@dataclass(frozen=True)
class StateAddress:
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: UUID | None


def parse_thread_id(config: ThreadConfig) -> str:
    thread_id = config["thread_id"]
    try:
        thread_id.encode()
    except UnicodeEncodeError as error:
        raise ValueError(f"thread_id {thread_id!r} has a surrogate code point, which storage cannot encode") from error
    return thread_id


def parse_checkpoint_id(config: ThreadConfig) -> UUID | None:
    match config.get("checkpoint_id"):
        case None:
            return None
        case UUID() as checkpoint_id:
            return checkpoint_id
        case str() as text:
            return UUID(text)
        case other:
            assert_never(other)


def parse_run_target(config: ThreadConfig) -> RunTarget:
    if config.get("checkpoint_ns"):
        raise ValueError("a run starts at the root graph, so its config cannot have a checkpoint_ns")
    return RunTarget(parse_thread_id(config), parse_checkpoint_id(config))


def parse_state_address(config: ThreadConfig) -> StateAddress:
    return StateAddress(parse_thread_id(config), config.get("checkpoint_ns", ""), parse_checkpoint_id(config))


def read_config() -> ThreadConfig:
    raise RuntimeError("read_config supplies a value only in a run of a graph that has a checkpointer")

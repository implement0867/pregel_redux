from dataclasses import dataclass
from typing import assert_never
from uuid import UUID


@dataclass(frozen=True)
class RunTarget:
    thread_id: str
    fork_from: UUID | None


@dataclass(frozen=True)
class StateAddress:
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: UUID | None


def parse_thread_id(thread_id: str) -> str:
    try:
        thread_id.encode()
    except UnicodeEncodeError as error:
        raise ValueError(f"thread_id {thread_id!r} has a surrogate code point, which storage cannot encode") from error
    return thread_id


def parse_checkpoint_id(checkpoint_id: UUID | None) -> UUID | None:
    match checkpoint_id:
        case None | UUID():
            return checkpoint_id
        case other:
            assert_never(other)


def parse_run_target(thread_id: str, fork_from: UUID | None) -> RunTarget:
    return RunTarget(parse_thread_id(thread_id), parse_checkpoint_id(fork_from))


def parse_state_address(thread_id: str, checkpoint_ns: str, checkpoint_id: UUID | None) -> StateAddress:
    return StateAddress(parse_thread_id(thread_id), checkpoint_ns, parse_checkpoint_id(checkpoint_id))

from dataclasses import dataclass
from typing import assert_never
from uuid import UUID


@dataclass(frozen=True)
class Segment:
    """Identifies one child graph run by the parent node and the task that runs the child graph."""

    node: str
    task_id: str


type Namespace = tuple[Segment, ...]


def encode_namespace(ns: Namespace) -> str:
    return "|".join(f"{segment.node}:{segment.task_id}" for segment in ns)


@dataclass(frozen=True)
class RunTarget:
    """Identifies the thread and the start checkpoint of a run.

    A node or a dependency of a `PersistentGraph` receives it through a parameter annotated `RunTarget`.

    Attributes
    ----------
    fork_from
        The checkpoint that the run continues from in a new branch of the thread.
        `None` selects the latest checkpoint.
    """

    thread_id: str
    fork_from: UUID | None


@dataclass(frozen=True)
class StateAddress:
    thread_id: str
    checkpoint_ns: Namespace
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


def parse_state_address(thread_id: str, checkpoint_ns: Namespace, checkpoint_id: UUID | None) -> StateAddress:
    return StateAddress(parse_thread_id(thread_id), checkpoint_ns, parse_checkpoint_id(checkpoint_id))

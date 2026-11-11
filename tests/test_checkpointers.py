import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import UUID

import pytest
from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import create_async_engine

from pregel_redux import (
    Checkpointer,
    Command,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    PersistentGraph,
    Reducer,
    StateGraph,
    interrupt,
)
from pregel_redux.checkpoints import RunResumeRecord
from pregel_redux.sqlite import SqliteCheckpointer
from tests.conftest import collect, join_lists


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class Output(TypedDict):
    notes: list[str]


def build_sqlite_checkpointer(path: Path) -> SqliteCheckpointer:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path / 'checkpoints.sqlite'}", poolclass=NullPool)
    checkpointer = SqliteCheckpointer(engine)
    asyncio.run(checkpointer.setup())
    return checkpointer


@pytest.fixture(params=["memory", "sqlite"])
def checkpointer(request: pytest.FixtureRequest, tmp_path: Path) -> Checkpointer:
    return InMemoryCheckpointer() if request.param == "memory" else build_sqlite_checkpointer(tmp_path)


def ask(state: State) -> State:
    return {"notes": [interrupt("question", response_schema=str)]}


def build_writer(name: str) -> Callable[[State], State]:
    return lambda state: {"notes": [name]}


def build_graph(checkpointer: Checkpointer) -> PersistentGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("first", build_writer("first"))
    builder.add_node("ask", ask)
    builder.add_node("last", build_writer("last"))
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "ask")
    builder.add_edge("ask", "last")
    return builder.compile(checkpointer=checkpointer)


def test_run_with_an_interrupt_continues_from_the_stored_checkpoint(checkpointer: Checkpointer) -> None:
    """
    Arrange: a thread that stopped at an interrupt, with its checkpoints in the given checkpointer.
    Act: run the thread with a resume value, with the sync function.
    Assert: the output has the notes of all nodes and the answer.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, thread_id="thread")

    actual = graph.invoke(Command(resume="answer"), thread_id="thread")

    expected = Finished(Output(notes=["first", "answer", "last"]))
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_threads_do_not_share_checkpoints(checkpointer: Checkpointer) -> None:
    """
    Arrange: two threads, of which one stopped at an interrupt.
    Act: read the latest checkpoint of a third thread and the steps of the two threads.
    Assert: the third thread has no checkpoint, and each thread has its own latest step.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, thread_id="stopped")
    graph.invoke({}, thread_id="finished")
    graph.invoke(Command(resume="answer"), thread_id="finished")

    missing = asyncio.run(checkpointer.aget_tuple("other", "", None))
    steps = [graph.get_state(thread_id).step for thread_id in ("stopped", "finished")]

    assert missing is None, f"Expected no checkpoint, got {missing}."
    assert steps == [1, 3], f"Expected the steps [1, 3], got {steps}."


@pytest.mark.parametrize(("skipped", "limit"), [(0, None), (0, 2), (1, None), (2, 1)])
def test_list_gives_the_checkpoints_before_an_id_with_the_newest_first(
    checkpointer: Checkpointer, skipped: int, limit: int | None
) -> None:
    """
    Arrange: a thread with a checkpoint for the input, the start and one node.
    Act: list the checkpoints before the id of one of them, with a limit.
    Assert: the list has the older checkpoints, the newest first, and not more than the limit.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, thread_id="thread")
    stored = asyncio.run(collect(checkpointer.alist("thread", "", before=None, limit=None)))
    before = None if skipped == 0 else stored[skipped - 1].checkpoint.id

    actual = asyncio.run(collect(checkpointer.alist("thread", "", before=before, limit=limit)))

    steps = [saved.checkpoint.step for saved in actual]
    expected = [1, 0, -1][skipped:][:limit]
    assert steps == expected, f"Expected the steps {expected}, got {steps}."


def test_checkpoint_is_found_by_its_id_with_its_pending_writes(checkpointer: Checkpointer) -> None:
    """
    Arrange: a thread that stopped at an interrupt and got a resume value two times.
    Act: read the checkpoint of the interrupt by its id, and read an id that the thread does not have.
    Assert: the checkpoint has one record for the interrupt and one for the last resume value.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, thread_id="thread")
    stopped = asyncio.run(collect(checkpointer.alist("thread", "", before=None, limit=1)))[0].checkpoint
    asyncio.run(checkpointer.aput_writes("thread", "", stopped.id, RunResumeRecord(value="one")))
    asyncio.run(checkpointer.aput_writes("thread", "", stopped.id, RunResumeRecord(value="two")))

    actual = asyncio.run(checkpointer.aget_tuple("thread", "", stopped.id))
    missing = asyncio.run(checkpointer.aget_tuple("thread", "", UUID(int=1)))

    writes = () if actual is None else actual.pending_writes
    kinds = sorted(write.kind.value for write in writes)
    resumes = [write.value for write in writes if isinstance(write, RunResumeRecord)]
    assert kinds == ["interrupt", "resume"], f"Expected one interrupt record and one resume record, got {kinds}."
    assert resumes == ["two"], f"Expected the last resume value, got {resumes}."
    assert missing is None, f"Expected no checkpoint for an unknown id, got {missing}."


def test_deleted_thread_has_no_checkpoints(checkpointer: Checkpointer) -> None:
    """
    Arrange: two threads with checkpoints.
    Act: delete one thread.
    Assert: the deleted thread has no checkpoint, and the other thread keeps its checkpoints.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, thread_id="deleted")
    graph.invoke({}, thread_id="kept")

    asyncio.run(checkpointer.adelete_thread("deleted"))

    deleted = asyncio.run(collect(checkpointer.alist("deleted", "", before=None, limit=None)))
    kept = asyncio.run(collect(checkpointer.alist("kept", "", before=None, limit=None)))
    assert deleted == [], f"Expected no checkpoints, got {deleted}."
    assert len(kept) == 3, f"Expected 3 checkpoints, got {len(kept)}."

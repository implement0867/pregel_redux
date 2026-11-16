import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import UUID

import pytest
from sqlalchemy import create_engine

from pregel_redux import (
    Checkpointer,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Message,
    PersistentGraph,
    Received,
    Reducer,
    StateGraph,
)
from pregel_redux.checkpoints import RunCommandRecord, SavedUpdates, SavedWrites
from pregel_redux.sqlite import SqliteCheckpointer
from pregel_redux.streams import UpdatesPart, WaitingPart
from tests.conftest import join_lists


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class Output(TypedDict):
    notes: list[str]


@pytest.fixture(params=["memory", "sqlite"])
def checkpointer(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Checkpointer]:
    if request.param == "memory":
        yield InMemoryCheckpointer()
        return
    engine = create_engine(f"sqlite:///{tmp_path / 'checkpoints.sqlite'}")
    sqlite_checkpointer = SqliteCheckpointer(engine)
    sqlite_checkpointer.setup()
    yield sqlite_checkpointer
    engine.dispose()


def ask(state: State, answer: Received[str]) -> State:
    return {"notes": [answer]}


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


def test_run_with_a_message_continues_from_the_stored_checkpoint(checkpointer: Checkpointer) -> None:
    """
    Arrange: a thread that waits for a message, with its checkpoints in the given checkpointer.
    Act: run the thread with a message, with the sync function.
    Assert: the output has the notes of all nodes and the answer.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, "thread")

    actual = graph.invoke(Message("answer"), "thread")

    expected = Finished(Output(notes=["first", "answer", "last"]))
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_threads_do_not_share_checkpoints(checkpointer: Checkpointer) -> None:
    """
    Arrange: two threads, of which one waits for a message.
    Act: read the latest checkpoint of a third thread and the steps of the two threads.
    Assert: the third thread has no checkpoint, and each thread has its own latest step.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, "waiting")
    graph.invoke({}, "finished")
    graph.invoke(Message("answer"), "finished")

    missing = checkpointer.get_tuple("other", "", None)
    steps = [graph.get_state(thread_id).step for thread_id in ("waiting", "finished")]

    assert missing is None, f"Expected no checkpoint, got {missing}."
    assert steps == [1, 4], f"Expected the steps [1, 4], got {steps}."


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
    graph.invoke({}, "thread")
    stored = list(checkpointer.list("thread", "", before=None, limit=None))
    before = None if skipped == 0 else stored[skipped - 1].checkpoint.id

    actual = list(checkpointer.list("thread", "", before=before, limit=limit))

    steps = [saved.checkpoint.step for saved in actual]
    expected = [1, 0, -1][skipped:][:limit]
    assert steps == expected, f"Expected the steps {expected}, got {steps}."


def test_checkpoint_is_found_by_its_id_with_its_pending_writes(checkpointer: Checkpointer) -> None:
    """
    Arrange: a thread that waits for a message and got a record of its first command two times.
    Act: read the newest checkpoint by its id, and read an id that the thread does not have.
    Assert: the checkpoint has only the last record of the command.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, "thread")
    stopped = next(checkpointer.list("thread", "", before=None, limit=1)).checkpoint
    one, two = (
        RunCommandRecord(
            index=0,
            command=SavedWrites(
                fields=SavedUpdates(updates={}, overwrites={}), branches=(node,), arrivals=(), sends=()
            ),
        )
        for node in ("one", "two")
    )
    checkpointer.put_writes("thread", "", stopped.id, one)
    checkpointer.put_writes("thread", "", stopped.id, two)

    actual = checkpointer.get_tuple("thread", "", stopped.id)
    missing = checkpointer.get_tuple("thread", "", UUID(int=1))

    assert actual is not None, "Expected a checkpoint for the id of the newest checkpoint, got None."
    assert actual.pending_writes == (two,), f"Expected only the last record of the command, got {actual}."
    assert missing is None, f"Expected no checkpoint for an unknown id, got {missing}."


def test_deleted_thread_has_no_checkpoints(checkpointer: Checkpointer) -> None:
    """
    Arrange: two threads with checkpoints.
    Act: delete one thread.
    Assert: the deleted thread has no checkpoint, and the other thread keeps its checkpoints.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, "deleted")
    graph.invoke({}, "kept")

    checkpointer.delete_thread("deleted")

    deleted = list(checkpointer.list("deleted", "", before=None, limit=None))
    kept = list(checkpointer.list("kept", "", before=None, limit=None))
    assert deleted == [], f"Expected no checkpoints, got {deleted}."
    assert len(kept) == 3, f"Expected 3 checkpoints, got {len(kept)}."


def test_sync_functions_of_the_graph_work_with_each_checkpointer(checkpointer: Checkpointer) -> None:
    """
    Arrange: a graph with a checkpointer.
    Act: run a thread until it waits, update its state, send a message and read its history, with the sync functions.
    Assert: the run finishes with the notes of the nodes, the update and the answer, and the history has each step.
    """
    graph = build_graph(checkpointer)
    stopped = list(graph.stream({}, "thread"))
    graph.update_state("thread", {"notes": ["update"]}, as_node="first")

    actual = graph.invoke(Message("answer"), "thread")

    steps = [snapshot.step for snapshot in graph.get_state_history("thread")]
    expected = Finished(Output(notes=["first", "update", "answer", "last"]))
    kinds = [type(part) for part in stopped if isinstance(part, UpdatesPart | WaitingPart)]
    assert kinds == [UpdatesPart, WaitingPart], (
        f"Expected the update of the first node and then the waiting part, got {stopped}."
    )
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert steps == [5, 4, 3, 2, 1, 0, -1], f"Expected the steps from 5 to -1, got {steps}."
    assert graph.get_state("thread").step == 5, "Expected the state of the last step."


def test_sync_functions_of_the_checkpointer_give_the_stored_checkpoints(checkpointer: Checkpointer) -> None:
    """
    Arrange: a thread with a checkpoint for the input, the start and one node.
    Act: read the newest checkpoint and the list with the sync functions, then delete the thread.
    Assert: the newest checkpoint is the first of the list, and the deleted thread has no checkpoint.
    """
    graph = build_graph(checkpointer)
    graph.invoke({}, "thread")

    newest = checkpointer.get_tuple("thread", "", None)
    listed = list(checkpointer.list("thread", "", before=None, limit=None))
    checkpointer.delete_thread("thread")

    steps = [saved.checkpoint.step for saved in listed]
    assert steps == [1, 0, -1], f"Expected the steps [1, 0, -1], got {steps}."
    assert newest == listed[0], f"Expected the newest checkpoint {listed[0]}, got {newest}."
    assert checkpointer.get_tuple("thread", "", None) is None, "Expected no checkpoint after the deletion."


def test_async_functions_of_the_graph_work_with_each_checkpointer(checkpointer: Checkpointer) -> None:
    """
    Arrange: a thread that waits for a message, made with the async function.
    Act: run the thread with a message, with the async function.
    Assert: the output has the notes of all nodes and the answer.
    """
    graph = build_graph(checkpointer)
    asyncio.run(graph.ainvoke({}, "thread"))

    actual = asyncio.run(graph.ainvoke(Message("answer"), "thread"))

    expected = Finished(Output(notes=["first", "answer", "last"]))
    assert actual == expected, f"Expected {expected}, got {actual}."

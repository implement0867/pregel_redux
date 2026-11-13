import asyncio
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, TypedDict, assert_never
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.pool import Pool, StaticPool

from pregel_redux import (
    Checkpointer,
    Endpoint,
    InMemoryCheckpointer,
    PersistentGraph,
    Reducer,
    RunAborted,
    ShutdownController,
    StateGraph,
)
from pregel_redux.checkpoints import RunCommandRecord, SavedWrites
from pregel_redux.sqlite import SqliteCheckpointer
from tests.conftest import join_lists


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


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


@pytest.fixture
def file_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{tmp_path / 'checkpoints.sqlite'}")
    yield engine
    engine.dispose()


def build_writer(name: str) -> Callable[[State], State]:
    return lambda state: {"notes": [name]}


def build_parallel_graph(checkpointer: Checkpointer, width: int) -> PersistentGraph[State, State, State]:
    builder = StateGraph(State)
    for index in range(width):
        builder.add_node(f"node {index}", build_writer(f"node {index}"))
        builder.add_edge(Endpoint.start, f"node {index}")
    return builder.compile(checkpointer=checkpointer)


@dataclass
class HeldCheckpointInsert:
    """Makes the worker thread that inserts the checkpoint with the given number wait until the test releases it."""

    held_number: int
    reached: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    inserts: int = 0

    def hold(self, **arguments: object) -> None:
        match arguments["statement"]:
            case str() as statement if statement.startswith("INSERT INTO checkpoints"):
                self.inserts += 1
                if self.inserts == self.held_number:
                    self.reached.set()
                    self.release.wait(timeout=10)
            case _:
                pass


class Stop(Enum):
    abort = "abort"
    cancel = "cancel"


@pytest.mark.parametrize("stop", list(Stop))
def test_thread_deleted_after_a_stop_in_a_checkpoint_save_has_no_checkpoint(file_engine: Engine, stop: Stop) -> None:
    """
    Arrange: a SQLite checkpointer whose worker thread waits in the insert of the checkpoint after the first node.
    Act: stop the async run while the insert waits, let the insert go on after a delay, and delete the thread
    after the run ends.
    Assert: the deleted thread has no checkpoint, because the run ends only after its insert.
    """
    checkpointer = SqliteCheckpointer(file_engine)
    checkpointer.setup()
    held = HeldCheckpointInsert(held_number=3)
    event.listen(file_engine, "before_cursor_execute", held.hold, named=True)
    graph = build_parallel_graph(checkpointer, 1)
    controller = ShutdownController()

    async def stop_and_delete() -> None:
        run = asyncio.create_task(graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal))
        try:
            await asyncio.to_thread(held.reached.wait, 10)
            asyncio.get_running_loop().call_later(0.1, held.release.set)
            match stop:
                case Stop.abort:
                    controller.abort()
                    with pytest.raises(RunAborted):
                        await run
                case Stop.cancel:
                    run.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await run
                case _:
                    assert_never(stop)
            await checkpointer.adelete_thread("thread")
        finally:
            held.release.set()

    asyncio.run(stop_and_delete())

    remaining = checkpointer.get_tuple("thread", "", None)
    assert remaining is None, f"Expected no checkpoint after the deletion, got {remaining}."


@pytest.mark.parametrize("pool", [None, StaticPool], ids=["connection-per-thread", "one-shared-connection"])
def test_in_memory_sqlite_database_is_rejected(pool: type[Pool] | None) -> None:
    """
    Arrange: an engine on an in-memory SQLite database, with the default pool or with one shared connection.
    Act: make a SQLite checkpointer on the engine.
    Assert: the checkpointer raises ValueError, because its worker threads cannot share that database safely.
    """
    engine = create_engine("sqlite://", poolclass=pool, connect_args={"check_same_thread": False})

    try:
        with pytest.raises(ValueError):
            SqliteCheckpointer(engine)
    finally:
        engine.dispose()


def test_history_with_a_negative_limit_is_rejected(checkpointer: Checkpointer) -> None:
    """
    Arrange: a thread with checkpoints.
    Act: read the history of the thread with the limit -1.
    Assert: the read raises ValueError on each checkpointer.
    """
    graph = build_parallel_graph(checkpointer, 1)
    graph.invoke({}, {"thread_id": "thread"})

    with pytest.raises(ValueError):
        list(graph.get_state_history({"thread_id": "thread"}, limit=-1))


def test_thread_id_with_a_lone_surrogate_is_rejected(checkpointer: Checkpointer) -> None:
    """
    Arrange: a graph of one node.
    Act: run a thread whose id has a lone surrogate, as json.loads gives for the escape \\udc80.
    Assert: the run raises ValueError on each checkpointer, and the thread has no checkpoint.
    """
    graph = build_parallel_graph(checkpointer, 1)

    with pytest.raises(ValueError):
        graph.invoke({}, {"thread_id": "\udc80"})


def test_pending_write_for_a_checkpoint_that_is_not_stored_is_refused(checkpointer: Checkpointer) -> None:
    """
    Arrange: a thread with checkpoints.
    Act: put a pending write for a checkpoint id that the thread does not have.
    Assert: the checkpointer raises KeyError, as InMemoryCheckpointer does.
    """
    build_parallel_graph(checkpointer, 1).invoke({}, {"thread_id": "thread"})
    record = RunCommandRecord(index=0, command=SavedWrites(fields={}, branches=("node 0",), arrivals=(), sends=()))

    with pytest.raises(KeyError):
        checkpointer.put_writes("thread", "", UUID(int=1), record)


def test_put_of_a_stored_checkpoint_id_replaces_the_checkpoint(checkpointer: Checkpointer) -> None:
    """
    Arrange: a thread with checkpoints.
    Act: put the newest checkpoint again with a different step.
    Assert: the checkpoint with that id has the new step, as on InMemoryCheckpointer.
    """
    build_parallel_graph(checkpointer, 1).invoke({}, {"thread_id": "thread"})
    newest = checkpointer.get_tuple("thread", "", None)
    if newest is None:
        raise LookupError("the thread has no checkpoint")
    changed = newest.checkpoint.model_copy(update={"step": 7})

    checkpointer.put("thread", changed)

    actual = checkpointer.get_tuple("thread", "", changed.id)
    step = None if actual is None else actual.checkpoint.step
    assert step == 7, f"Expected the step 7, got {step}."

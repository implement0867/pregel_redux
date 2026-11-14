import asyncio
import signal
import threading
import time
from collections.abc import AsyncIterator
from typing import Annotated, TypedDict

import pytest

from pregel_redux import (
    Command,
    Depends,
    Endpoint,
    InMemoryCheckpointer,
    Reducer,
    RunAborted,
    Scope,
    Send,
    ShutdownController,
    StateGraph,
    get_stream_writer,
    graceful_shutdown,
)
from pregel_redux.streams import ChildPart, TaskResultPart, TaskStartPart
from tests.conftest import join_lists


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


@pytest.mark.parametrize("sync", [False, True])
def test_custom_part_of_a_sync_node_reaches_the_reader_while_the_node_runs(*, sync: bool) -> None:
    """
    Arrange: a sync node that pauses until the event loop of the run is idle, writes a custom part,
    and then waits until the reader confirms that it received the part.
    Act: read the custom stream of a run with the async function or the sync function, and confirm each part.
    Assert: the node sees the confirmation before its wait ends.
    """
    confirmed = threading.Event()
    seen: list[bool] = []

    def write_and_wait(state: State) -> State:
        time.sleep(0.1)
        get_stream_writer()("progress")
        seen.append(confirmed.wait(timeout=1))
        return {"notes": ["done"]}

    builder = StateGraph(State)
    builder.add_node("work", write_and_wait)
    builder.add_edge(Endpoint.start, "work")
    graph = builder.compile()

    async def read() -> None:
        async for _ in graph.astream({}):
            confirmed.set()

    if sync:
        for _ in graph.stream({}):
            confirmed.set()
    else:
        asyncio.run(read())

    assert seen == [True], (
        f"Expected the reader to receive the custom part while the node runs, got {seen}. "
        "A part that waits for the end of its node gives no progress."
    )


def test_keyboard_interrupt_in_the_reader_of_a_sync_stream_stops_the_run() -> None:
    """
    Arrange: a chain of two sync nodes where the first node sends SIGINT to the thread that reads the stream.
    Act: read the sync stream of the run.
    Assert: the reader gets KeyboardInterrupt and the second node does not run, as with invoke.
    """
    reader = threading.get_ident()
    calls: list[str] = []

    def interrupt_the_reader(state: State) -> State:
        calls.append("first")
        signal.pthread_kill(reader, signal.SIGINT)
        return {"notes": ["first"]}

    def write_second(state: State) -> State:
        calls.append("second")
        return {"notes": ["second"]}

    builder = StateGraph(State)
    builder.add_node("first", interrupt_the_reader)
    builder.add_node("second", write_second)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "second")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    earlier = signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        with pytest.raises(KeyboardInterrupt):
            list(graph.stream({}, {"thread_id": "thread"}))
    finally:
        signal.signal(signal.SIGINT, earlier)

    assert calls == ["first"], f"Expected the run to stop when the reader stops, got the calls {calls}."


def test_abort_after_a_request_with_a_grace_period_runs_the_whole_cleanup() -> None:
    """
    Arrange: a node with a run-scoped provider whose cleanup takes longer than a grace period, where the node
    requests a graceful shutdown with that grace period, lets the run react, aborts the run and waits.
    Act: run the thread with the signal.
    Assert: the run raises RunAborted after the whole cleanup.
    """
    controller = ShutdownController()
    events: list[str] = []

    async def open_link() -> AsyncIterator[str]:
        events.append("open")
        try:
            yield "link"
        finally:
            await asyncio.sleep(0.2)
            events.append("close")

    async def abort_and_wait(state: State, link: Annotated[str, Depends(open_link, scope=Scope.run)]) -> State:
        controller.shutdown(grace_period=0.05)
        await asyncio.sleep(0)
        controller.abort()
        await asyncio.Event().wait()
        return {"notes": [link]}

    builder = StateGraph(State)
    builder.add_node("wait", abort_and_wait)
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(RunAborted):
        asyncio.run(graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal))

    assert events == ["open", "close"], (
        f"Expected the whole cleanup after the abort, got {events}. "
        "The timer of the grace period must not cancel the cleanup of an abort."
    )


def test_grace_period_that_ends_in_the_cleanup_of_a_graceful_stop_does_not_cut_the_cleanup() -> None:
    """
    Arrange: a chain of two nodes where the first node has a run-scoped provider whose cleanup takes longer than
    a grace period, and requests a graceful shutdown with that grace period.
    Act: run the thread with the signal.
    Assert: the cleanup of the provider finishes.
    """
    controller = ShutdownController()
    events: list[str] = []

    async def open_link() -> AsyncIterator[str]:
        events.append("open")
        try:
            yield "link"
        finally:
            await asyncio.sleep(0.2)
            events.append("close")

    async def request_shutdown(state: State, link: Annotated[str, Depends(open_link, scope=Scope.run)]) -> State:
        controller.shutdown(grace_period=0.05)
        return {"notes": [link]}

    builder = StateGraph(State)
    builder.add_node("first", request_shutdown)
    builder.add_node("second", lambda state: {"notes": ["second"]})
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "second")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    async def run() -> object:
        try:
            return await graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal)
        except RunAborted as error:
            return error

    outcome = asyncio.run(run())

    assert events == ["open", "close"], (
        f"Expected the whole cleanup after a stop that ended all its nodes in the grace period, got {events}. "
        f"The run gave {outcome!r}."
    )


def test_graceful_shutdown_that_cannot_bind_a_signal_keeps_the_earlier_handlers() -> None:
    """
    Arrange: the handler of SIGINT before the block.
    Act: enter graceful_shutdown with SIGINT and SIGKILL, which no process can handle.
    Assert: the block raises OSError, and SIGINT keeps its earlier handler.
    """
    handler_before = signal.getsignal(signal.SIGINT)

    try:
        with pytest.raises(OSError), graceful_shutdown(signals=(signal.SIGINT, signal.SIGKILL)):
            pass
        handler_after = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, handler_before)

    assert handler_after is handler_before, (
        f"Expected the earlier handler {handler_before!r} after the failed block, got {handler_after!r}. "
        "A handler of a controller that no caller holds makes the process ignore SIGINT."
    )


def test_tasks_of_two_sends_to_a_child_graph_have_different_ids_and_namespaces() -> None:
    """
    Arrange: a graph without a checkpointer whose first node sends two inputs to a child graph of one node.
    Act: stream a run.
    Assert: the start parts of the two inner tasks have different ids and different namespaces.
    """
    child = StateGraph(State)
    child.add_node("inner", lambda state: {"notes": ["inner"]})
    child.add_edge(Endpoint.start, "inner")
    builder = StateGraph(State)
    builder.add_node(
        "route", lambda state: Command[str](goto=[Send("child", {"notes": ["a"]}), Send("child", {"notes": ["b"]})])
    )
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "route")
    graph = builder.compile()

    parts = list(graph.stream({}))

    inner = [
        (part.ns, part.part.id)
        for part in parts
        if isinstance(part, ChildPart) and isinstance(part.part, TaskStartPart) and part.part.name == "inner"
    ]
    ids = {task_id for _, task_id in inner}
    namespaces = {ns for ns, _ in inner}
    assert len(inner) == 2, f"Expected two inner tasks, got {inner}."
    assert len(ids) == 2, f"Expected a different id for each inner task, got {inner}."
    assert len(namespaces) == 2, f"Expected a different namespace for each child graph, got {inner}."


def test_stream_gives_no_task_result_for_a_child_graph_that_a_graceful_shutdown_stopped() -> None:
    """
    Arrange: a child graph of two nodes where the first node requests a graceful shutdown.
    Act: stream a run with the signal.
    Assert: the stream has no result part that reports the unfinished child graph task as a task with an update.
    """
    controller = ShutdownController()

    def write_first(state: State) -> State:
        controller.shutdown()
        return {"notes": ["first"]}

    child = StateGraph(State)
    child.add_node("first", write_first)
    child.add_node("second", lambda state: {"notes": ["second"]})
    child.add_edge(Endpoint.start, "first")
    child.add_edge("first", "second")
    builder = StateGraph(State)
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "child")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    parts = list(graph.stream({}, {"thread_id": "thread"}, signal=controller.signal))
    snapshot = graph.get_state({"thread_id": "thread"})

    results = [part for part in parts if isinstance(part, TaskResultPart) and part.name == "child"]
    assert snapshot.next == ("child",), f"Expected the child graph task to wait, got {snapshot.next}."
    assert results == [], (
        f"Expected no result for the child graph task that did not finish, got {results}. "
        "The part has no update and no interrupt, the same as a task that ended with an empty update."
    )

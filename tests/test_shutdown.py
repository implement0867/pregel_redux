import asyncio
import signal
import threading
from collections.abc import AsyncIterator
from typing import Annotated, TypedDict, assert_never

import pytest

from pregel_redux import (
    Depends,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupted,
    Reducer,
    RunAborted,
    ShutdownController,
    ShutdownSignal,
    StateGraph,
    StopReason,
    StreamMode,
    graceful_shutdown,
    interrupt,
    read_shutdown,
)
from pregel_redux.streams import (
    CheckpointPart,
    CustomPart,
    InterruptsPart,
    StreamPart,
    TaskResultPart,
    TaskStartPart,
    UpdatesPart,
    ValuesPart,
)
from tests.conftest import collect, join_lists


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


def test_graceful_shutdown_stops_the_run_after_the_current_step() -> None:
    """
    Arrange: a chain of two nodes where the first node requests a graceful shutdown.
    Act: run the thread with the signal, and run the thread again without an input.
    Assert: the first run stops with the write of the first node, and each node ran one time.
    """
    controller = ShutdownController()
    calls: list[str] = []

    def write_first(state: State) -> State:
        calls.append("first")
        controller.shutdown()
        return {"notes": ["first"]}

    def write_second(state: State) -> State:
        calls.append("second")
        return {"notes": ["second"]}

    builder = StateGraph(State)
    builder.add_node("first", write_first)
    builder.add_node("second", write_second)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "second")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    stopped = graph.invoke({}, {"thread_id": "thread"}, signal=controller.signal)
    finished = graph.invoke(None, {"thread_id": "thread"})

    assert stopped == Interrupted((), {"notes": ["first"]}, StopReason.shutdown), (
        f"Expected a stop after the first node, got {stopped}."
    )
    assert finished == Finished(State(notes=["first", "second"])), f"Expected the two writes, got {finished}."
    assert calls == ["first", "second"], f"Expected each node to run one time, got {calls}."


def test_graceful_shutdown_lets_the_other_nodes_of_the_step_finish() -> None:
    """
    Arrange: a step with a node that requests a graceful shutdown and a node that pauses before its write.
    Act: run the thread with the signal, and run the thread again without an input.
    Assert: the stop has the writes of the two nodes, and the second run does not run the two nodes again.
    """
    controller = ShutdownController()
    calls: list[str] = []

    async def request_shutdown(state: State) -> State:
        calls.append("request")
        controller.shutdown()
        return {"notes": ["request"]}

    async def write_slowly(state: State) -> State:
        calls.append("slow")
        for _ in range(3):
            await asyncio.sleep(0)
        return {"notes": ["slow"]}

    def write_last(state: State) -> State:
        calls.append("last")
        return {"notes": ["last"]}

    builder = StateGraph(State)
    builder.add_node("request", request_shutdown)
    builder.add_node("slow", write_slowly)
    builder.add_node("last", write_last)
    builder.add_edge(Endpoint.start, "request")
    builder.add_edge(Endpoint.start, "slow")
    builder.add_edge(("request", "slow"), "last")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    stopped = asyncio.run(graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal))
    finished = asyncio.run(graph.ainvoke(None, {"thread_id": "thread"}))

    assert stopped == Interrupted((), {"notes": ["request", "slow"]}, StopReason.shutdown), (
        f"Expected the two writes, got {stopped}."
    )
    assert finished == Finished(State(notes=["request", "slow", "last"])), f"Expected three writes, got {finished}."
    assert sorted(calls) == ["last", "request", "slow"], f"Expected each node to run one time, got {calls}."


def test_graceful_shutdown_stops_a_child_graph_after_its_current_step() -> None:
    """
    Arrange: a child graph of two nodes where the first node requests a graceful shutdown.
    Act: run the thread with the signal, and run the thread again without an input.
    Assert: the first run stops before the second node of the child graph, and each node ran one time.
    """
    controller = ShutdownController()
    calls: list[str] = []

    def write_first(state: State) -> State:
        calls.append("first")
        controller.shutdown()
        return {"notes": ["first"]}

    def write_second(state: State) -> State:
        calls.append("second")
        return {"notes": ["second"]}

    child = StateGraph(State)
    child.add_node("first", write_first)
    child.add_node("second", write_second)
    child.add_edge(Endpoint.start, "first")
    child.add_edge("first", "second")
    builder = StateGraph(State)
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "child")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    stopped = graph.invoke({}, {"thread_id": "thread"}, signal=controller.signal)
    calls_at_the_stop = list(calls)
    finished = graph.invoke(None, {"thread_id": "thread"})

    assert stopped == Interrupted((), {"notes": []}, StopReason.shutdown), (
        f"Expected a stop inside the child graph, got {stopped}."
    )
    assert calls_at_the_stop == ["first"], f"Expected only the first node before the stop, got {calls_at_the_stop}."
    assert finished == Finished(State(notes=["first", "second"])), f"Expected the two writes, got {finished}."
    assert calls == ["first", "second"], f"Expected each node to run one time, got {calls}."


def test_run_with_a_signal_that_already_requests_a_shutdown_runs_no_node() -> None:
    """
    Arrange: a graph of one node and a controller with a graceful shutdown request.
    Act: run the thread with the signal.
    Assert: the run stops with the input and the node did not run.
    """
    controller = ShutdownController()
    controller.shutdown()
    calls: list[str] = []

    def write(state: State) -> State:
        calls.append("write")
        return {"notes": ["write"]}

    builder = StateGraph(State)
    builder.add_node("write", write)
    builder.add_edge(Endpoint.start, "write")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    stopped = graph.invoke({"notes": ["input"]}, {"thread_id": "thread"}, signal=controller.signal)

    assert stopped == Interrupted((), {"notes": ["input"]}, StopReason.shutdown), (
        f"Expected a stop with the input, got {stopped}."
    )
    assert calls == [], f"Expected no node to run, got {calls}."


def test_graceful_shutdown_before_a_step_reports_the_questions_that_wait() -> None:
    """
    Arrange: a thread where a node waits for the answer to a question, and a controller with a shutdown request.
    Act: run the thread again without an input and with the signal.
    Assert: the stop has the question that waits, the same as the state of the thread.
    """
    controller = ShutdownController()
    controller.shutdown()

    def ask(state: State) -> State:
        return {"notes": [interrupt("question", response_schema=str)]}

    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    asked = graph.invoke({}, {"thread_id": "thread"})

    stopped = graph.invoke(None, {"thread_id": "thread"}, signal=controller.signal)

    waiting = graph.get_state({"thread_id": "thread"}).interrupts
    assert isinstance(asked, Interrupted), f"Expected a question in the first run, got {asked}."
    assert isinstance(stopped, Interrupted), f"Expected a stop, got {stopped}."
    assert stopped.interrupts == asked.interrupts == waiting, f"Expected the question {waiting}, got {stopped}."


def test_abort_stops_a_node_at_once_and_runs_the_cleanup() -> None:
    """
    Arrange: a node with a provider that has a cleanup, where the node aborts the run and then waits.
    Act: run the thread with the signal, and run the thread again without an input.
    Assert: the first run raises RunAborted after the cleanup, and the second run runs the node again.
    """
    controller = ShutdownController()
    events: list[str] = []

    async def open_link() -> AsyncIterator[str]:
        events.append("open")
        try:
            yield "link"
        finally:
            events.append("close")

    async def wait_or_write(state: State, link: Annotated[str, Depends(open_link)]) -> State:
        events.append("node")
        if events.count("node") == 1:
            controller.abort()
            await asyncio.Event().wait()
        return {"notes": [link]}

    builder = StateGraph(State)
    builder.add_node("wait", wait_or_write)
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(RunAborted):
        asyncio.run(graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal))
    events_at_the_abort = list(events)
    finished = asyncio.run(graph.ainvoke(None, {"thread_id": "thread"}))

    assert events_at_the_abort == ["open", "node", "close"], f"Expected the cleanup, got {events_at_the_abort}."
    assert finished == Finished(State(notes=["link"])), f"Expected the write of the second run, got {finished}."


def test_abort_from_a_different_thread_stops_the_run() -> None:
    """
    Arrange: an async node that starts a thread that aborts the run, and then waits.
    Act: run the thread with the signal.
    Assert: the run raises RunAborted.
    """
    controller = ShutdownController()

    async def wait(state: State) -> State:
        threading.Thread(target=controller.abort).start()
        await asyncio.Event().wait()
        return {}

    builder = StateGraph(State)
    builder.add_node("wait", wait)
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(RunAborted):
        asyncio.run(graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal))


def test_run_with_a_signal_that_is_already_aborted_raises_without_a_checkpoint() -> None:
    """
    Arrange: a graph of one node and a controller that is aborted.
    Act: run the thread with the signal.
    Assert: the run raises RunAborted and the thread has no checkpoint.
    """
    controller = ShutdownController()
    controller.abort()
    builder = StateGraph(State)
    builder.add_node("write", lambda state: {"notes": ["write"]})
    builder.add_edge(Endpoint.start, "write")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(RunAborted):
        graph.invoke({}, {"thread_id": "thread"}, signal=controller.signal)

    history = list(graph.get_state_history({"thread_id": "thread"}))
    assert history == [], f"Expected no checkpoint, got {history}."


def test_grace_period_aborts_a_node_that_does_not_finish() -> None:
    """
    Arrange: a node that requests a graceful shutdown with a short grace period and then waits.
    Act: run the thread with the signal.
    Assert: the run raises RunAborted.
    """
    controller = ShutdownController()

    async def wait(state: State) -> State:
        controller.shutdown(grace_period=0.01)
        await asyncio.Event().wait()
        return {}

    builder = StateGraph(State)
    builder.add_node("wait", wait)
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(RunAborted):
        asyncio.run(graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal))


def test_node_that_finishes_in_the_grace_period_gives_a_graceful_stop() -> None:
    """
    Arrange: a chain of two nodes where the first node requests a graceful shutdown with a long grace period.
    Act: run the thread with the signal.
    Assert: the run stops with the write of the first node.
    """
    controller = ShutdownController()

    def write_first(state: State) -> State:
        controller.shutdown(grace_period=60)
        return {"notes": ["first"]}

    builder = StateGraph(State)
    builder.add_node("first", write_first)
    builder.add_node("second", lambda state: {"notes": ["second"]})
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "second")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    stopped = graph.invoke({}, {"thread_id": "thread"}, signal=controller.signal)

    assert stopped == Interrupted((), {"notes": ["first"]}, StopReason.shutdown), (
        f"Expected a stop after the first node, got {stopped}."
    )


def test_node_reads_the_shutdown_signal_of_its_run() -> None:
    """
    Arrange: a node that requests the shutdown signal as a dependency and writes its state before and after a request.
    Act: run the thread with the signal, and run a second thread without a signal.
    Assert: the node sees the request of its controller, and a run without a signal has a signal with no request.
    """
    controller = ShutdownController()
    seen: list[tuple[bool, bool]] = []

    def read(state: State, shutdown: Annotated[ShutdownSignal, Depends(read_shutdown)]) -> State:
        before = shutdown.requested
        controller.shutdown()
        seen.append((before, shutdown.requested))
        return {}

    builder = StateGraph(State)
    builder.add_node("read", read)
    builder.add_edge(Endpoint.start, "read")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    graph.invoke({}, {"thread_id": "with signal"}, signal=controller.signal)
    graph.invoke({}, {"thread_id": "without signal"})

    assert seen == [(False, True), (False, False)], f"Expected the request only in the first run, got {seen}."


def test_stream_ends_after_the_current_step_at_a_graceful_shutdown() -> None:
    """
    Arrange: a chain of two nodes where the first node requests a graceful shutdown.
    Act: read the stream of updates of a run with the signal.
    Assert: the stream has the update of the first node and then the stop.
    """
    controller = ShutdownController()

    def write_first(state: State) -> State:
        controller.shutdown()
        return {"notes": ["first"]}

    builder = StateGraph(State)
    builder.add_node("first", write_first)
    builder.add_node("second", lambda state: {"notes": ["second"]})
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "second")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    parts = list(graph.stream({}, {"thread_id": "thread"}, stream_mode=StreamMode.updates, signal=controller.signal))

    def read_node(part: StreamPart) -> str | None:
        match part:
            case UpdatesPart(node=node):
                return node
            case ValuesPart() | InterruptsPart() | CustomPart() | CheckpointPart() | TaskStartPart() | TaskResultPart():
                return None
            case _:
                assert_never(part)

    names = [read_node(part) for part in parts]
    assert names == ["first", None], f"Expected the update of the first node and the stop, got {parts}."


def test_stream_raises_at_an_abort() -> None:
    """
    Arrange: a node that aborts the run and then waits.
    Act: read the async stream of a run with the signal.
    Assert: the stream raises RunAborted.
    """
    controller = ShutdownController()

    async def wait(state: State) -> State:
        controller.abort()
        await asyncio.Event().wait()
        return {}

    builder = StateGraph(State)
    builder.add_node("wait", wait)
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(RunAborted):
        asyncio.run(
            collect(
                graph.astream({}, {"thread_id": "thread"}, stream_mode=StreamMode.updates, signal=controller.signal)
            )
        )


def test_shutdown_request_after_an_abort_keeps_the_abort() -> None:
    """
    Arrange: a controller that is aborted.
    Act: request a graceful shutdown.
    Assert: the signal is requested and aborted.
    """
    controller = ShutdownController()
    controller.abort()

    controller.shutdown()

    assert controller.signal.requested, "Expected a requested signal."
    assert controller.signal.aborted, "Expected the abort to stay."


def test_signal_is_aborted_when_the_clock_reaches_the_deadline() -> None:
    """
    Arrange: a controller with a clock that the test moves, and a shutdown request with a grace period.
    Act: read the signal before the deadline and at the deadline.
    Assert: the signal is requested and not aborted before the deadline, and aborted at the deadline.
    """
    now = [100.0]
    controller = ShutdownController(clock=lambda: now[0])
    controller.shutdown(grace_period=5.0)

    before = (controller.signal.requested, controller.signal.aborted, controller.signal.seconds_until_abort())
    now[0] = 105.0
    after = (controller.signal.requested, controller.signal.aborted, controller.signal.seconds_until_abort())

    assert before == (True, False, 5.0), f"Expected a request with 5 seconds left, got {before}."
    assert after == (True, True, 0.0), f"Expected an abort at the deadline, got {after}."


def test_earlier_deadline_wins_between_two_shutdown_requests() -> None:
    """
    Arrange: a controller with a clock that the test moves, and a shutdown request with a long grace period.
    Act: request a shutdown with a shorter grace period, then one with a longer grace period, then one with none.
    Assert: the time until the abort is the shortest grace period after each request.
    """
    controller = ShutdownController(clock=lambda: 100.0)
    controller.shutdown(grace_period=10.0)

    controller.shutdown(grace_period=2.0)
    after_shorter = controller.signal.seconds_until_abort()
    controller.shutdown(grace_period=30.0)
    after_longer = controller.signal.seconds_until_abort()
    controller.shutdown()
    after_none = controller.signal.seconds_until_abort()

    assert (after_shorter, after_longer, after_none) == (2.0, 2.0, 2.0), (
        f"Expected 2 seconds after each request, got {(after_shorter, after_longer, after_none)}."
    )


def test_signal_without_a_deadline_has_no_time_until_the_abort() -> None:
    """
    Arrange: a controller.
    Act: read the time until the abort before a request, after a request with no grace period, and after an abort.
    Assert: there is no time before the abort, and the time is zero after it.
    """
    controller = ShutdownController()

    before = controller.signal.seconds_until_abort()
    controller.shutdown()
    requested = controller.signal.seconds_until_abort()
    controller.abort()
    aborted = controller.signal.seconds_until_abort()

    assert (before, requested, aborted) == (None, None, 0.0), (
        f"Expected no time, no time and zero, got {(before, requested, aborted)}."
    )


def test_first_interrupt_signal_requests_a_graceful_shutdown() -> None:
    """
    Arrange: a chain of two nodes where the first node raises SIGINT.
    Act: inside graceful_shutdown, run the thread.
    Assert: the run stops after the first node, and the earlier handler is restored after the block.
    """
    handler_before = signal.getsignal(signal.SIGINT)

    async def raise_signal(state: State) -> State:
        signal.raise_signal(signal.SIGINT)
        return {"notes": ["first"]}

    builder = StateGraph(State)
    builder.add_node("first", raise_signal)
    builder.add_node("second", lambda state: {"notes": ["second"]})
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "second")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with graceful_shutdown() as shutdown:
        stopped = asyncio.run(graph.ainvoke({}, {"thread_id": "thread"}, signal=shutdown))

    assert stopped == Interrupted((), {"notes": ["first"]}, StopReason.shutdown), (
        f"Expected a stop after the first node, got {stopped}."
    )
    assert signal.getsignal(signal.SIGINT) is handler_before, "Expected the earlier handler after the block."


def test_second_interrupt_signal_aborts_the_run() -> None:
    """
    Arrange: a node that raises SIGINT two times and then waits.
    Act: inside graceful_shutdown, run the thread.
    Assert: the run raises RunAborted.
    """

    async def raise_two_times_and_wait(state: State) -> State:
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)
        await asyncio.Event().wait()
        return {}

    builder = StateGraph(State)
    builder.add_node("wait", raise_two_times_and_wait)
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with graceful_shutdown() as shutdown, pytest.raises(RunAborted):
        asyncio.run(graph.ainvoke({}, {"thread_id": "thread"}, signal=shutdown))


def test_terminate_signal_requests_a_graceful_shutdown_with_the_grace_period() -> None:
    """
    Arrange: a node that raises SIGTERM and then waits.
    Act: inside graceful_shutdown with a short grace period, run the thread.
    Assert: the run raises RunAborted.
    """

    async def raise_and_wait(state: State) -> State:
        signal.raise_signal(signal.SIGTERM)
        await asyncio.Event().wait()
        return {}

    builder = StateGraph(State)
    builder.add_node("wait", raise_and_wait)
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with graceful_shutdown(grace_period=0.01) as shutdown, pytest.raises(RunAborted):
        asyncio.run(graph.ainvoke({}, {"thread_id": "thread"}, signal=shutdown))


def test_caller_that_cancels_a_run_with_a_signal_gets_its_own_cancellation() -> None:
    """
    Arrange: a node that waits, in a run with a signal that has no request.
    Act: run the thread with a timeout of the caller.
    Assert: the caller gets TimeoutError and not RunAborted.
    """
    controller = ShutdownController()

    async def wait(state: State) -> State:
        await asyncio.Event().wait()
        return {}

    builder = StateGraph(State)
    builder.add_node("wait", wait)
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(TimeoutError):
        asyncio.run(
            asyncio.wait_for(graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal), timeout=0.01)
        )

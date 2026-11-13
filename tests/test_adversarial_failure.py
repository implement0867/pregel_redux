import asyncio
import threading
from collections.abc import Iterator
from typing import Annotated, TypedDict

import pytest
from pydantic import BaseModel, JsonValue, ValidationError

from pregel_redux import (
    Command,
    Depends,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupted,
    Reducer,
    RunAborted,
    Scope,
    Send,
    ShutdownController,
    StateGraph,
    ThreadConfig,
    interrupt,
)
from tests.conftest import join_lists, read_error


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class Decision(BaseModel):
    approved: bool


def test_timeout_that_expires_during_the_cleanup_of_a_sync_provider_raises_timeout_error() -> None:
    """
    Arrange: a graph without a checkpointer whose node has a sync generator provider, and a timeout around the run
    that expires while the cleanup of the provider runs in its thread.
    Act: run the graph inside the timeout.
    Assert: the timeout raises TimeoutError, as for a cancellation at any other moment of the run.
    """
    released = threading.Event()
    deadlines: list[tuple[asyncio.AbstractEventLoop, asyncio.Timeout]] = []

    def expire_the_timeout() -> None:
        loop, deadline = deadlines[0]
        deadline.reschedule(loop.time())
        loop.call_soon(released.set)

    def open_link() -> Iterator[str]:
        yield "link"
        deadlines[0][0].call_soon_threadsafe(expire_the_timeout)
        released.wait(timeout=5)

    def use_link(state: State, link: Annotated[str, Depends(open_link)]) -> State:
        return State(notes=[link])

    builder = StateGraph(State)
    builder.add_node("use", use_link)
    builder.add_edge(Endpoint.start, "use")
    graph = builder.compile()

    async def run_with_a_timeout() -> None:
        async with asyncio.timeout(None) as deadline:
            deadlines.append((asyncio.get_running_loop(), deadline))
            await graph.ainvoke({})

    with pytest.raises(TimeoutError):
        asyncio.run(run_with_a_timeout())


@pytest.mark.parametrize("scope", list(Scope), ids=lambda scope: scope.name)
def test_cleanup_error_of_a_sync_provider_keeps_its_own_context_when_another_node_fails(scope: Scope) -> None:
    """
    Arrange: a step with a node that fails, and a node with a sync generator provider whose cleanup fails
    while it handles an earlier error.
    Act: run the graph.
    Assert: the context chain of the raised cleanup error has the earlier error of the cleanup and then the error
    of the failed node, as it has for an async provider.
    """

    def open_link() -> Iterator[str]:
        try:
            yield "link"
        finally:
            try:
                raise OSError("socket closed")
            finally:
                raise LookupError("cleanup failed")

    def use_link(state: State, link: Annotated[str, Depends(open_link, scope=scope)]) -> State:
        return State(notes=[link])

    def raise_error(state: State) -> State:
        raise ValueError("node failed")

    builder = StateGraph(State)
    builder.add_node("use", use_link)
    builder.add_node("fail", raise_error)
    builder.add_edge(Endpoint.start, "use")
    builder.add_edge(Endpoint.start, "fail")
    with pytest.raises(LookupError) as raised:
        asyncio.run(builder.compile().ainvoke({}))

    chain: list[BaseException] = [raised.value]
    while chain[-1].__context__ is not None and len(chain) < 10:
        chain.append(chain[-1].__context__)
    actual = [type(error).__name__ for error in chain]
    expected = ["LookupError", "OSError", "ValueError"]
    assert actual == expected, f"Expected the context chain {expected}, got {actual}."


@pytest.mark.parametrize("with_checkpointer", [False, True], ids=["without-checkpointer", "with-checkpointer"])
def test_run_scoped_provider_that_suppresses_the_exception_of_its_node_is_rejected(*, with_checkpointer: bool) -> None:
    """
    Arrange: a node that fails, with a run-scoped generator provider that suppresses the exception of the node.
    Act: run the graph.
    Assert: the run raises a RuntimeError, as it does for a node-scoped provider that suppresses the exception.
    """

    def open_client() -> Iterator[str]:
        try:
            yield "client"
        except LookupError:
            return

    def raise_error(state: State, client: Annotated[str, Depends(open_client, scope=Scope.run)]) -> State:
        raise LookupError(client)

    builder = StateGraph(State)
    builder.add_node("use", raise_error)
    builder.add_edge(Endpoint.start, "use")

    with pytest.raises(RuntimeError):
        if with_checkpointer:
            asyncio.run(builder.compile(checkpointer=InMemoryCheckpointer()).ainvoke({}, {"thread_id": "thread"}))
        else:
            asyncio.run(builder.compile().ainvoke({}))


def test_answer_that_is_not_of_the_response_schema_leaves_the_question_waiting() -> None:
    """
    Arrange: a thread that waits for the answer to a question whose response schema is a Pydantic model.
    Act: answer with a value that is not of the schema, then answer the same question by its id with a value
    of the schema.
    Assert: the question still waits after the rejected answer, and the second answer finishes the run.
    """

    def decide(state: State) -> State:
        decision = interrupt("approve?", response_schema=Decision)
        return State(notes=["approved" if decision.approved else "rejected"])

    builder = StateGraph(State)
    builder.add_node("decide", decide)
    builder.add_edge(Endpoint.start, "decide")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    asked = asyncio.run(graph.ainvoke({}, config))
    questions = asked.interrupts if isinstance(asked, Interrupted) else ()
    with pytest.raises(ValidationError):
        asyncio.run(graph.ainvoke(Command(resume="text"), config))

    waiting = asyncio.run(graph.aget_state(config)).interrupts
    answers: dict[str, JsonValue] = {item.id: {"approved": True} for item in questions}
    actual = asyncio.run(graph.ainvoke(Command[str](resume_by_interrupt_id=answers), config))

    assert waiting == questions, f"Expected the question {questions} to wait after the rejected answer, got {waiting}."
    assert actual == Finished(State(notes=["approved"])), f"Expected the second answer to finish the run, got {actual}."


def test_deferred_node_runs_after_the_nodes_of_the_pending_sends() -> None:
    """
    Arrange: a router that sends two payloads to a worker, and a deferred node that the start reaches.
    Act: run the graph.
    Assert: the deferred node runs after the two workers and sees their writes.
    """

    def route(state: State) -> Command[str]:
        return Command(update=State(notes=["route"]), goto=[Send("work", State(notes=[topic])) for topic in "ab"])

    def work(state: State) -> State:
        return State(notes=[f"work {state.get('notes', [])[0]}"])

    def late(state: State) -> State:
        return State(notes=[f"late saw {len(state.get('notes', []))}"])

    builder = StateGraph(State)
    builder.add_node("route", route)
    builder.add_node("work", work)
    builder.add_node("late", late, defer=True)
    builder.add_edge(Endpoint.start, "route")
    builder.add_edge(Endpoint.start, "late")

    actual = asyncio.run(builder.compile().ainvoke({}))

    expected = State(notes=["route", "work a", "work b", "late saw 3"])
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_abort_after_an_answer_keeps_the_error_of_the_task() -> None:
    """
    Arrange: a node that fails before its question in the first run, and that receives a plain answer to its
    question in the second run and then waits until the run is aborted.
    Act: run the thread, run it with the answer and the signal, and read the state after the abort.
    Assert: the task keeps the error of the first run, because it did not finish and did not ask a question.
    """
    controller = ShutdownController()
    runs: list[str] = []

    async def fail_then_ask(state: State) -> State:
        runs.append("run")
        if len(runs) == 1:
            raise ConnectionError("first run")
        answer = interrupt("question", response_schema=str)
        controller.abort()
        await asyncio.Event().wait()
        return State(notes=[answer])

    builder = StateGraph(State)
    builder.add_node("ask", fail_then_ask)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({}, config))
    with pytest.raises(RunAborted):
        asyncio.run(graph.ainvoke(Command(resume="yes"), config, signal=controller.signal))

    errors = [read_error(task) for task in asyncio.run(graph.aget_state(config)).tasks]

    assert errors == ["ConnectionError('first run')"], f"Expected the error of the first run, got {errors}."


def test_graceful_stop_of_a_child_graph_keeps_the_error_of_its_task() -> None:
    """
    Arrange: a step with a node and a child graph that both fail, and a second run in which the node requests
    a graceful shutdown before the child graph runs its failed node again.
    Act: run the thread, then run it again with the signal, and read the state with the child graphs.
    Assert: the task of the child graph keeps its error, because it did not finish and did not ask a question,
    and the state of the child graph still has the error of its node.
    """
    controller = ShutdownController()
    runs: list[str] = []

    async def stop_or_fail(state: State) -> State:
        runs.append("stop")
        if runs.count("stop") == 1:
            raise ValueError("first run")
        controller.shutdown()
        return State(notes=["stop"])

    def fail_one_time(state: State) -> State:
        runs.append("fail")
        if runs.count("fail") == 1:
            raise ConnectionError("no network")
        return State(notes=["fail"])

    child = StateGraph(State)
    child.add_node("fail", fail_one_time)
    child.add_edge(Endpoint.start, "fail")
    builder = StateGraph(State)
    builder.add_node("a_stop", stop_or_fail)
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "a_stop")
    builder.add_edge(Endpoint.start, "child")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    with pytest.raises(ExceptionGroup):
        asyncio.run(graph.ainvoke({}, config))

    stopped = asyncio.run(graph.ainvoke(None, config, signal=controller.signal))
    state = asyncio.run(graph.aget_state(config, subgraphs=True))

    task = next(task for task in state.tasks if task.name == "child")
    inner = [] if task.state is None else [read_error(item) for item in task.state.tasks]
    assert isinstance(stopped, Interrupted), f"Expected a graceful stop, got {stopped}."
    assert runs.count("fail") == 1, f"Expected the failed node of the child graph not to run again, got {runs}."
    assert inner == ["ConnectionError('no network')"], f"Expected the error in the child graph, got {inner}."
    assert read_error(task) == "ConnectionError('no network')", f"Expected the error of the child task, got {task}."

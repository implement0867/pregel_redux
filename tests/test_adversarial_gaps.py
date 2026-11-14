import asyncio
import signal
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, TypedDict

import pytest
from pydantic import JsonValue
from sqlalchemy import create_engine

from pregel_redux import (
    Checkpointer,
    Command,
    Depends,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupted,
    Reducer,
    RunAborted,
    Send,
    ShutdownController,
    StateGraph,
    ThreadConfig,
    graceful_shutdown,
    interrupt,
)
from pregel_redux.snapshots import FinishedTask
from pregel_redux.sqlite import SqliteCheckpointer
from pregel_redux.streams import ChildPart, ValuesPart
from tests.conftest import join_lists, read_error


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


def build_writer(name: str) -> Callable[[State], State]:
    def write(state: State) -> State:
        return State(notes=[name])

    return write


def ask(state: State) -> State:
    return State(notes=[interrupt("question", response_schema=str)])


def build_parent_of_a_child_graph() -> StateGraph[State, State, State]:
    child = StateGraph(State)
    child.add_node("first", build_writer("first"))
    child.add_node("second", build_writer("second"))
    child.add_edge(Endpoint.start, "first")
    child.add_edge("first", "second")
    builder = StateGraph(State)
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "child")
    return builder


def test_values_parts_of_a_child_graph_have_the_namespace_of_the_child_graph() -> None:
    """
    Arrange: a parent graph whose only node is a child graph of two nodes.
    Act: stream a run.
    Assert: the values parts of the parent graph have only the states of the parent graph, and the states of
    the child graph are in child parts with the namespace of the child graph.
    """
    graph = build_parent_of_a_child_graph().compile()

    parts = list(graph.stream({}))

    outer = [part.values for part in parts if isinstance(part, ValuesPart)]
    inner = [
        (part.ns[0].split(":")[0], part.part.values)
        for part in parts
        if isinstance(part, ChildPart) and isinstance(part.part, ValuesPart)
    ]
    expected_outer = [State(notes=[]), State(notes=["first", "second"])]
    expected_inner = [
        ("child", State(notes=[])),
        ("child", State(notes=["first"])),
        ("child", State(notes=["first", "second"])),
    ]
    assert outer == expected_outer, f"Expected the root parts {expected_outer}, got {outer}."
    assert inner == expected_inner, f"Expected the child graph parts {expected_inner}, got {inner}."


def test_command_with_a_send_keeps_the_send_tasks_that_wait_for_an_answer() -> None:
    """
    Arrange: a thread in which two Send tasks wait for an answer to their question.
    Act: run the thread with a Command that sends a payload to a writer, then answer the two questions.
    Assert: the two questions wait before and after the Command, and the last run finishes with the note of
    the writer and the two answers.
    """

    def route(state: State) -> Command[str]:
        return Command(goto=[Send("ask", {}), Send("ask", {})])

    builder = StateGraph(State)
    builder.add_node("route", route)
    builder.add_node("ask", ask)
    builder.add_node("write", build_writer("write"))
    builder.add_edge(Endpoint.start, "route")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    graph.invoke({}, config)
    before = graph.get_state(config).interrupts

    graph.invoke(Command(goto=[Send("write", {})]), config)
    after = graph.get_state(config).interrupts
    answers: dict[str, JsonValue] = {item.id: f"answer {index}" for index, item in enumerate(before)}
    actual = graph.invoke(Command[str](resume_by_interrupt_id=answers), config)

    expected = Finished(State(notes=["answer 0", "answer 1", "write"]))
    assert len(before) == 2, f"Expected two waiting questions before the Command, got {before}."
    assert after == before, f"Expected the questions {before} to wait after the Command, got {after}."
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_command_with_an_update_keeps_a_deferred_node_after_the_node_that_waits_for_an_answer() -> None:
    """
    Arrange: a thread in which a node waits for an answer to its question, and a deferred node that the start
    also reaches.
    Act: run the thread with a Command that has only an update, then answer the question.
    Assert: the deferred node does not run in the run of the Command, and it runs after the answer and sees
    the note of the node that waited.
    """
    calls: list[str] = []

    def write_late(state: State) -> State:
        calls.append("late")
        return State(notes=[f"late saw {state.get('notes')}"])

    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_node("late", write_late, defer=True)
    builder.add_edge(Endpoint.start, "ask")
    builder.add_edge(Endpoint.start, "late")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    graph.invoke({}, config)

    graph.invoke(Command(update=State(notes=["update"])), config)
    calls_of_the_command = list(calls)
    waiting = graph.get_state(config).next
    actual = graph.invoke(Command(resume="yes"), config)

    expected = Finished(State(notes=["update", "yes", "late saw ['update', 'yes']"]))
    assert calls_of_the_command == [], (
        f"Expected the deferred node not to run in the run of the Command, got {calls_of_the_command}."
    )
    assert waiting == ("ask",), f"Expected only the waiting node next after the Command, got {waiting}."
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_task_that_finished_after_a_failure_has_a_result_and_no_error() -> None:
    """
    Arrange: a step of a node that fails in the first run and a node that asks a question.
    Act: run the thread, run it again so that the failed node finishes while the other node waits, and read
    the state.
    Assert: the task of the node that finished has its result and no error.
    """
    failures: list[str] = []

    def fail_one_time(state: State) -> State:
        if not failures:
            failures.append("failed")
            raise ConnectionError("no network")
        return State(notes=["recovered"])

    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_node("recover", fail_one_time)
    builder.add_edge(Endpoint.start, "ask")
    builder.add_edge(Endpoint.start, "recover")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    with pytest.raises(ConnectionError):
        graph.invoke({}, config)

    graph.invoke(None, config)
    state = graph.get_state(config)

    actual = [(task.name, task.status) for task in state.tasks if task.name == "recover"]
    expected = [("recover", FinishedTask({"notes": ["recovered"]}))]
    assert actual == expected, f"Expected the finished task {expected}, got {actual}."


def test_state_update_without_a_node_after_two_nodes_of_one_step_is_rejected() -> None:
    """
    Arrange: a thread that finished after two nodes ran in the same step.
    Act: update the state without as_node.
    Assert: the update raises ValueError, because the node that made the last update is not clear.
    """
    builder = StateGraph(State)
    builder.add_node("left", build_writer("left"))
    builder.add_node("right", build_writer("right"))
    builder.add_edge(Endpoint.start, "left")
    builder.add_edge(Endpoint.start, "right")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    graph.invoke({}, config)

    with pytest.raises(ValueError):
        graph.update_state(config, State(notes=["update"]))


@pytest.mark.parametrize("provider_kind", ["generator", "function"])
def test_abort_during_the_setup_of_a_sync_provider_stops_the_sync_node_before_it_runs(provider_kind: str) -> None:
    """
    Arrange: a step of a sync node with a sync provider and an async node that waits. The setup of the provider
    aborts the run and then waits until the cancellation reached the nodes of the step.
    Act: run the thread with the signal.
    Assert: the run raises RunAborted, the cancellation reached the step during the setup, and the function of
    the sync node did not run.
    """
    controller = ShutdownController()
    cancelled = threading.Event()
    handed_over: list[bool] = []
    calls: list[str] = []

    def abort_during_the_setup() -> str:
        controller.abort()
        handed_over.append(cancelled.wait(timeout=5))
        return "link"

    def open_link_with_cleanup() -> Iterator[str]:
        yield abort_during_the_setup()

    provider = open_link_with_cleanup if provider_kind == "generator" else abort_during_the_setup

    def use_link(state: State, link: Annotated[str, Depends(provider)]) -> State:
        calls.append(link)
        return State(notes=[link])

    async def wait_for_the_cancellation(state: State) -> State:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return State(notes=["waited"])

    builder = StateGraph(State)
    builder.add_node("use", use_link)
    builder.add_node("wait", wait_for_the_cancellation)
    builder.add_edge(Endpoint.start, "use")
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    async def run_with_a_guard() -> object:
        async with asyncio.timeout(10):
            return await graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal)

    with pytest.raises(RunAborted):
        asyncio.run(run_with_a_guard())

    assert handed_over == [True], (
        f"Expected the cancellation to reach the step during the setup of the provider, got {handed_over}."
    )
    assert calls == [], f"Expected the sync node not to run after the abort, got the calls {calls}."


def test_checkpoint_that_is_stored_again_after_the_delete_of_its_thread_has_no_pending_writes(
    checkpointer: Checkpointer,
) -> None:
    """
    Arrange: a thread that stopped at an interrupt, so that its newest checkpoint has a pending write.
    Act: delete the thread, store the same checkpoint again, and read it.
    Assert: the checkpoint that was read has no pending writes.
    """
    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({}, {"thread_id": "thread"})
    stopped = checkpointer.get_tuple("thread", "", None)
    assert stopped is not None, "Expected a checkpoint of the thread that stopped at an interrupt, got None."
    assert stopped.pending_writes, f"Expected a pending write before the delete, got {stopped.pending_writes}."

    checkpointer.delete_thread("thread")
    checkpointer.put("thread", stopped.checkpoint)
    stored_again = checkpointer.get_tuple("thread", "", stopped.checkpoint.id)

    assert stored_again is not None, "Expected the checkpoint that was stored again, got None."
    assert stored_again.checkpoint == stopped.checkpoint, (
        f"Expected the checkpoint {stopped.checkpoint}, got {stored_again.checkpoint}."
    )
    assert stored_again.pending_writes == (), (
        f"Expected no pending writes after the delete of the thread, got {stored_again.pending_writes}."
    )


def test_graceful_shutdown_block_that_raises_restores_the_earlier_handlers() -> None:
    """
    Arrange: the handlers of SIGINT and SIGTERM before the block.
    Act: leave a graceful_shutdown block with an exception.
    Assert: the handlers inside the block are not the earlier handlers, and after the block SIGINT and SIGTERM
    have their earlier handlers.
    """
    numbers = (signal.SIGINT, signal.SIGTERM)
    handlers_before = [signal.getsignal(number) for number in numbers]
    handlers_inside: list[object] = []

    try:
        with pytest.raises(LookupError), graceful_shutdown():
            handlers_inside.extend(signal.getsignal(number) for number in numbers)
            raise LookupError("the block failed")
        handlers_after = [signal.getsignal(number) for number in numbers]
    finally:
        signal.signal(signal.SIGINT, handlers_before[0])
        signal.signal(signal.SIGTERM, handlers_before[1])

    replaced = [inside is not before for inside, before in zip(handlers_inside, handlers_before, strict=True)]
    assert replaced == [True, True], f"Expected new handlers inside the block, got {handlers_inside}."
    assert handlers_after == handlers_before, (
        f"Expected the earlier handlers {handlers_before} after the failed block, got {handlers_after}."
    )


def test_grace_period_of_a_second_shutdown_request_aborts_a_run_that_does_not_stop() -> None:
    """
    Arrange: a node that requests a graceful shutdown without a grace period, requests it again with a short
    grace period, and then waits for an event that nothing sets.
    Act: run the thread with the signal, inside a timeout that guards the test.
    Assert: the run raises RunAborted at the end of the grace period.
    """
    controller = ShutdownController()

    async def request_two_times_and_wait(state: State) -> State:
        controller.shutdown()
        controller.shutdown(grace_period=0.05)
        await asyncio.Event().wait()
        return State(notes=["waited"])

    builder = StateGraph(State)
    builder.add_node("wait", request_two_times_and_wait)
    builder.add_edge(Endpoint.start, "wait")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    async def run_with_a_guard() -> object:
        async with asyncio.timeout(5):
            return await graph.ainvoke({}, {"thread_id": "thread"}, signal=controller.signal)

    with pytest.raises(RunAborted):
        asyncio.run(run_with_a_guard())


def test_answer_of_a_node_that_an_abort_cancelled_reaches_the_node_in_the_next_run() -> None:
    """
    Arrange: an async node that asks a question in each run, and that in its second run aborts the run after
    it received the answer and then waits.
    Act: run the thread, run it with the answer and the signal, and run it again with no input.
    Assert: the second run raises RunAborted, and the third run finishes with the answer of the second run.
    """
    controller = ShutdownController()
    runs: list[str] = []

    async def ask_then_wait(state: State) -> State:
        runs.append("run")
        answer = interrupt("question", response_schema=str)
        if len(runs) == 2:
            controller.abort()
            await asyncio.Event().wait()
        return State(notes=[answer])

    builder = StateGraph(State)
    builder.add_node("ask", ask_then_wait)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")

    async def run_with_a_guard(command: Command[str] | None) -> object:
        async with asyncio.timeout(5):
            return await graph.ainvoke(command, config, signal=controller.signal if command else None)

    asyncio.run(graph.ainvoke({}, config))
    with pytest.raises(RunAborted):
        asyncio.run(run_with_a_guard(Command(resume="yes")))

    actual = asyncio.run(run_with_a_guard(None))

    expected = Finished(State(notes=["yes"]))
    assert actual == expected, f"Expected the saved answer to finish the third run with {expected}, got {actual}."


def test_answer_of_a_node_that_failed_stays_and_its_later_question_waits_after_another_failure() -> None:
    """
    Arrange: a node with two questions that fails between the questions in its second run and its fourth run.
    Act: run the thread, run it with the first answer, run it two times with no input, read the state, and
    run it with the second answer.
    Assert: the third run asks only the second question, the state after the fourth run has the error of
    that run and only the second question, and the fifth run finishes with the two answers.
    """
    runs: list[str] = []

    def ask_two_questions(state: State) -> State:
        runs.append("run")
        first = interrupt("first question", response_schema=str)
        if len(runs) in (2, 4):
            raise ConnectionError(f"run {len(runs)}")
        second = interrupt("second question", response_schema=str)
        return State(notes=[first, second])

    builder = StateGraph(State)
    builder.add_node("ask", ask_two_questions)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    graph.invoke({}, config)
    with pytest.raises(ConnectionError):
        graph.invoke(Command(resume="one"), config)

    third = graph.invoke(None, config)
    assert isinstance(third, Interrupted), f"Expected the third run to stop at a question, got {third}."
    asked = [item.value for item in third.interrupts]
    assert asked == ["second question"], f"Expected the third run to ask only the second question, got {asked}."
    with pytest.raises(ConnectionError):
        graph.invoke(None, config)
    state = graph.get_state(config)
    actual = graph.invoke(Command(resume="two"), config)

    waiting = [item.value for item in state.interrupts]
    errors = [read_error(task) for task in state.tasks]
    expected = Finished(State(notes=["one", "two"]))
    assert waiting == ["second question"], f"Expected only the second question to wait, got {waiting}."
    assert errors == ["ConnectionError('run 4')"], f"Expected the error of the fourth run, got {errors}."
    assert actual == expected, f"Expected {expected}, got {actual}."

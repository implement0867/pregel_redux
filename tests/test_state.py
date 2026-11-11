import asyncio
from collections.abc import Callable
from typing import Annotated, TypedDict

import pytest

from pregel_redux import (
    Command,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupted,
    PersistentGraph,
    Reducer,
    StateGraph,
    interrupt,
)
from pregel_redux.checkpoints import CheckpointSource
from tests.conftest import collect, join_lists


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class Output(TypedDict):
    notes: list[str]


def build_writer(name: str, calls: list[str]) -> Callable[[State], State]:
    def write(state: State) -> State:
        calls.append(name)
        return {"notes": [name]}

    return write


def ask(state: State) -> State:
    return {"notes": [interrupt("question", response_schema=str)]}


def build_chain(
    calls: list[str],
    *,
    interrupt_before: tuple[str, ...] | str = (),
    interrupt_after: tuple[str, ...] | str = (),
) -> PersistentGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    for name in ("first", "second", "third"):
        builder.add_node(name, build_writer(name, calls))
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", "third")
    return builder.compile(
        checkpointer=InMemoryCheckpointer(),
        interrupt_before="*" if interrupt_before == "*" else tuple(interrupt_before),
        interrupt_after="*" if interrupt_after == "*" else tuple(interrupt_after),
    )


def build_asking_graph() -> PersistentGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("first", build_writer("first", []))
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "ask")
    return builder.compile(checkpointer=InMemoryCheckpointer())


def test_state_of_a_finished_thread_has_its_values_and_no_next_node() -> None:
    """
    Arrange: a thread of a run that finished.
    Act: read the state of the thread, with the sync function and with the async function.
    Assert: the state has the values of the output, no next node, and the step of the last node.
    """
    graph = build_chain([])
    graph.invoke({"notes": ["input"]}, thread_id="thread")

    actual = graph.get_state("thread")
    from_async = asyncio.run(graph.aget_state("thread"))

    summary = (actual.values, actual.next, actual.step, actual.source)
    expected = ({"notes": ["input", "first", "second", "third"]}, (), 3, CheckpointSource.loop)
    assert summary == expected, f"Expected {expected}, got {summary}."
    assert from_async == actual, f"Expected the same state from the async function, got {from_async}."


def test_state_of_a_thread_at_an_interrupt_has_the_node_and_the_interrupt() -> None:
    """
    Arrange: a thread that stopped at an interrupt.
    Act: read the state of the thread.
    Assert: the next node is the node that called interrupt, and the state has the value of the interrupt.
    """
    graph = build_asking_graph()
    graph.invoke({}, thread_id="thread")

    actual = graph.get_state("thread")

    values = [item.value for item in actual.interrupts]
    assert actual.next == ("ask",), f"Expected the next node 'ask', got {actual.next}."
    assert values == ["question"], f"Expected the value of the interrupt, got {values}."


def test_state_of_a_thread_without_a_checkpoint_is_rejected() -> None:
    """
    Arrange: a graph with a checkpointer that has no thread.
    Act: read the state of a thread.
    Assert: the call raises LookupError that names the thread.
    """
    graph = build_chain([])

    with pytest.raises(LookupError, match="thread 'missing'"):
        graph.get_state("missing")


@pytest.mark.parametrize(("skipped", "limit"), [(0, None), (2, None), (0, 2), (1, 1)])
def test_state_history_gives_the_states_before_an_id_with_the_newest_first(skipped: int, limit: int | None) -> None:
    """
    Arrange: a thread of a run that finished.
    Act: read the history of the thread before the id of one state, with a limit.
    Assert: the history has the older states with the newest first, and each state names the node that runs next.
    """
    graph = build_chain([])
    graph.invoke({}, thread_id="thread")
    history = list(graph.get_state_history("thread"))
    before = None if skipped == 0 else history[skipped - 1].checkpoint_id

    actual = asyncio.run(collect(graph.aget_state_history("thread", before=before, limit=limit)))

    nodes = [state.next for state in actual]
    expected = [(), ("third",), ("second",), ("first",), ("__start__",)][skipped:][:limit]
    assert nodes == expected, f"Expected the next nodes {expected}, got {nodes}."


def test_run_from_an_earlier_checkpoint_runs_the_later_nodes_again() -> None:
    """
    Arrange: a thread of a run that finished, and the id of the state before the second node.
    Act: run the thread without an input from that checkpoint.
    Assert: only the later nodes run, the output is equal, and the new checkpoints follow the old checkpoint.
    """
    calls: list[str] = []
    graph = build_chain(calls)
    expected = graph.invoke({}, thread_id="thread")
    earlier = next(state for state in graph.get_state_history("thread") if state.next == ("second",))
    calls.clear()

    actual = graph.invoke(None, thread_id="thread", checkpoint_id=earlier.checkpoint_id)

    history = list(graph.get_state_history("thread"))
    fork = next(state for state in history if state.source is CheckpointSource.fork)
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == ["second", "third"], f"Expected the calls ['second', 'third'], got {calls}."
    assert fork.parent_id == earlier.checkpoint_id, f"Expected a fork of {earlier.checkpoint_id}, got {fork}."
    assert history[0].next == (), f"Expected the newest state to be the end of the new run, got {history[0]}."


def test_run_from_a_checkpoint_that_the_thread_does_not_have_is_rejected() -> None:
    """
    Arrange: two threads of runs that finished.
    Act: run one thread from a checkpoint of the other thread.
    Assert: the run raises LookupError.
    """
    graph = build_chain([])
    graph.invoke({}, thread_id="thread")
    graph.invoke({}, thread_id="other")
    other_id = graph.get_state("other").checkpoint_id

    with pytest.raises(LookupError, match="no checkpoint"):
        graph.invoke(None, thread_id="thread", checkpoint_id=other_id)


@pytest.mark.parametrize("as_node", ["first", None])
def test_state_update_counts_as_the_output_of_a_node(as_node: str | None) -> None:
    """
    Arrange: a thread that stopped before the second node.
    Act: update the state as the first node, by name or by inference, then continue the run.
    Assert: the new state has the update and the second node next, and the run finishes with the update.
    """
    calls: list[str] = []
    graph = build_chain(calls, interrupt_before=("second",))
    graph.invoke({}, thread_id="thread")

    checkpoint_id = graph.update_state("thread", {"notes": ["edit"]}, as_node=as_node)

    state = graph.get_state("thread")
    summary = (state.checkpoint_id, state.values, state.next, state.source)
    expected = (checkpoint_id, {"notes": ["first", "edit"]}, ("second",), CheckpointSource.update)
    actual = graph.invoke(None, thread_id="thread")
    assert summary == expected, f"Expected {expected}, got {summary}."
    assert actual == Finished(Output(notes=["first", "edit", "second", "third"])), f"Got {actual}."
    assert calls == ["first", "second", "third"], f"Expected each node to run one time, got {calls}."


def test_state_update_as_a_later_node_skips_the_nodes_before_it() -> None:
    """
    Arrange: a thread that stopped before the second node.
    Act: update the state as the second node, then continue the run.
    Assert: the second node does not run, and the third node runs.
    """
    calls: list[str] = []
    graph = build_chain(calls, interrupt_before=("second",))
    graph.invoke({}, thread_id="thread")

    asyncio.run(graph.aupdate_state("thread", {"notes": ["edit"]}, as_node="second"))

    actual = graph.invoke(None, thread_id="thread")
    assert actual == Finished(Output(notes=["first", "edit", "third"])), f"Got {actual}."
    assert calls == ["first", "third"], f"Expected the calls ['first', 'third'], got {calls}."


def test_state_update_of_a_new_thread_starts_the_thread() -> None:
    """
    Arrange: a graph with a checkpointer that has no thread.
    Act: update the state of a thread as the first node, then continue the run.
    Assert: the run starts at the second node with the update.
    """
    calls: list[str] = []
    graph = build_chain(calls)

    graph.update_state("thread", {"notes": ["edit"]}, as_node="first")

    actual = graph.invoke(None, thread_id="thread")
    assert actual == Finished(Output(notes=["edit", "second", "third"])), f"Got {actual}."
    assert calls == ["second", "third"], f"Expected the calls ['second', 'third'], got {calls}."


@pytest.mark.parametrize(
    ("as_node", "started", "message"),
    [(None, False, "give as_node"), ("missing", True, "not a node"), ("missing", False, "not a node")],
)
def test_state_update_without_a_clear_node_is_rejected(as_node: str | None, *, started: bool, message: str) -> None:
    """
    Arrange: a graph with three nodes, and a thread with or without checkpoints.
    Act: update the state with no node that the graph can infer, or as a node that the graph does not have.
    Assert: the call raises ValueError with the reason.
    """
    graph = build_chain([])
    if started:
        graph.invoke({}, thread_id="thread")

    with pytest.raises(ValueError, match=message):
        graph.update_state("thread", {"notes": ["edit"]}, as_node=as_node)


def test_state_update_from_an_earlier_checkpoint_becomes_the_newest_state() -> None:
    """
    Arrange: a thread of a run that finished, and the id of the state before the third node.
    Act: update the state from that checkpoint as the second node, then continue the run.
    Assert: the run continues from the update, so only the third node runs.
    """
    calls: list[str] = []
    graph = build_chain(calls)
    graph.invoke({}, thread_id="thread")
    earlier = next(state for state in graph.get_state_history("thread") if state.next == ("third",))
    calls.clear()

    graph.update_state("thread", {"notes": ["edit"]}, checkpoint_id=earlier.checkpoint_id)

    actual = graph.invoke(None, thread_id="thread")
    assert actual == Finished(Output(notes=["first", "second", "edit", "third"])), f"Got {actual}."
    assert calls == ["third"], f"Expected the calls ['third'], got {calls}."


@pytest.mark.parametrize(
    ("before", "after", "expected_stops"),
    [
        (("second",), (), [["first"]]),
        ((), ("second",), [["first", "second"]]),
        (("second",), ("second",), [["first"], ["first", "second"]]),
        ("*", (), [[], ["first"], ["first", "second"]]),
        ((), ("third",), [["first", "second", "third"]]),
    ],
)
def test_run_stops_before_and_after_the_named_nodes(
    before: tuple[str, ...] | str, after: tuple[str, ...] | str, expected_stops: list[list[str]]
) -> None:
    """
    Arrange: a graph of three nodes in a chain that stops before or after named nodes.
    Act: run a thread, then continue it without an input until it finishes.
    Assert: each stop has no interrupt value and happens after the expected calls, and each node runs one time.
    """
    calls: list[str] = []
    graph = build_chain(calls, interrupt_before=before, interrupt_after=after)
    stops: list[list[str]] = []

    result = graph.invoke({}, thread_id="thread")
    while isinstance(result, Interrupted):
        assert result.interrupts == (), f"Expected a stop without an interrupt value, got {result}."
        stops.append([*calls])
        result = graph.invoke(None, thread_id="thread")

    assert stops == expected_stops, f"Expected the stops {expected_stops}, got {stops}."
    assert result == Finished(Output(notes=["first", "second", "third"])), f"Got {result}."


def test_new_input_on_a_stopped_thread_stops_before_the_named_node_again() -> None:
    """
    Arrange: a thread that stopped before the second node.
    Act: run the thread with a new input.
    Assert: the run stops before the second node again.
    """
    calls: list[str] = []
    graph = build_chain(calls, interrupt_before=("second",))
    graph.invoke({}, thread_id="thread")

    actual = graph.invoke({"notes": ["more"]}, thread_id="thread")

    assert actual == Interrupted(()), f"Expected a stop, got {actual}."
    assert calls == ["first", "first"], f"Expected the first node to run two times, got {calls}."


def test_child_graph_stops_before_its_named_node_and_continues() -> None:
    """
    Arrange: a parent graph whose child graph stops before its second node.
    Act: run a thread, then continue it without an input.
    Assert: the first run stops, and the second run finishes with each node called one time.
    """
    calls: list[str] = []
    child = StateGraph(State, output_schema=State)
    child.add_node("first", build_writer("first", calls))
    child.add_node("second", build_writer("second", calls))
    child.add_edge(Endpoint.start, "first")
    child.add_edge("first", "second")
    parent = StateGraph(State, output_schema=Output)
    parent.add_node("child", child)
    parent.add_edge(Endpoint.start, "child")
    graph = parent.compile(checkpointer=InMemoryCheckpointer(), interrupt_before=["child"])

    stopped = [graph.invoke({}, thread_id="thread")]
    actual = graph.invoke(None, thread_id="thread")

    assert stopped == [Interrupted(())], f"Expected one stop, got {stopped}."
    assert calls == ["first", "second"], f"Expected each node to run one time, got {calls}."
    assert isinstance(actual, Finished), f"Expected a finished run, got {actual}."


def test_unknown_node_in_interrupt_before_is_rejected() -> None:
    """
    Arrange: a builder with one node.
    Act: compile it with interrupt_after that names another node.
    Assert: compile raises ValueError that names the node.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("first", build_writer("first", []))
    builder.add_edge(Endpoint.start, "first")

    with pytest.raises(ValueError, match="missing"):
        builder.compile(checkpointer=InMemoryCheckpointer(), interrupt_after=["missing"])


def test_resume_value_continues_a_thread_that_stopped_before_a_node() -> None:
    """
    Arrange: a graph that stops before a node that calls interrupt.
    Act: run a thread, continue it without an input, then give a resume value.
    Assert: the runs stop without a value, stop with the value of the interrupt, then finish with the answer.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer(), interrupt_before=["ask"])

    first = graph.invoke({}, thread_id="thread")
    second = graph.invoke(None, thread_id="thread")
    third = graph.invoke(Command(resume="answer"), thread_id="thread")

    values = [item.value for item in second.interrupts] if isinstance(second, Interrupted) else None
    assert first == Interrupted(()), f"Expected a stop without a value, got {first}."
    assert values == ["question"], f"Expected the value of the interrupt, got {second}."
    assert third == Finished(Output(notes=["answer"])), f"Expected the answer, got {third}."

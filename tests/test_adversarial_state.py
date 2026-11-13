from collections.abc import Callable
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest

from pregel_redux import (
    Command,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    PersistentGraph,
    Reducer,
    StateGraph,
    ThreadConfig,
    interrupt,
)
from tests.conftest import join_lists


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


def build_writer(name: str) -> Callable[[State], State]:
    def write(state: State) -> State:
        return {"notes": [name]}

    return write


def build_asker(name: str) -> Callable[[State], State]:
    def ask(state: State) -> State:
        return {"notes": [f"{name} got {interrupt(f'{name} question', response_schema=str)}"]}

    return ask


def build_failing_once(name: str, failures: list[str]) -> Callable[[State], State]:
    def fail_once(state: State) -> State:
        if not failures:
            failures.append(name)
            raise ConnectionError(name)
        return {"notes": [name]}

    return fail_once


def test_answer_by_id_with_the_checkpoint_id_of_the_stop_reaches_the_waiting_node() -> None:
    """
    Arrange: a thread that stopped at an interrupt, and the state of the thread with its checkpoint id.
    Act: run the thread from that checkpoint id with an answer by the id of the waiting interrupt.
    Assert: the run finishes with the answer, as the same command without the checkpoint id does.
    """
    builder = StateGraph(State)
    builder.add_node("ask", build_asker("ask"))
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, {"thread_id": "thread"})
    stopped = graph.get_state({"thread_id": "thread"})
    answers = {item.id: "yes" for item in stopped.interrupts}

    actual = graph.invoke(
        Command(resume_by_interrupt_id=answers), {"thread_id": "thread", "checkpoint_id": stopped.checkpoint_id}
    )

    expected = Finished(State(notes=["ask got yes"]))
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_plain_answer_with_the_checkpoint_id_of_two_waiting_interrupts_is_rejected() -> None:
    """
    Arrange: a thread in which two nodes wait for an answer, and the checkpoint id of the thread.
    Act: run the thread from that checkpoint id with one plain answer.
    Assert: the run raises ValueError, as the same command without the checkpoint id does.
    """
    builder = StateGraph(State)
    builder.add_node("left", build_asker("left"))
    builder.add_node("right", build_asker("right"))
    builder.add_edge(Endpoint.start, "left")
    builder.add_edge(Endpoint.start, "right")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, {"thread_id": "thread"})
    stopped = graph.get_state({"thread_id": "thread"})

    with pytest.raises(ValueError, match="resume_by_interrupt_id"):
        graph.invoke(Command(resume="yes"), {"thread_id": "thread", "checkpoint_id": stopped.checkpoint_id})


def run_with_an_input(graph: PersistentGraph[State, State, State], config: ThreadConfig) -> object:
    return graph.invoke({}, config)


def update_as_the_node(graph: PersistentGraph[State, State, State], config: ThreadConfig) -> object:
    return graph.update_state(config, State(notes=["edit"]), as_node="write")


@pytest.mark.parametrize("call", [run_with_an_input, update_as_the_node])
def test_unknown_checkpoint_id_of_a_thread_without_checkpoints_is_rejected(
    call: Callable[[PersistentGraph[State, State, State], ThreadConfig], object],
) -> None:
    """
    Arrange: a graph with a checkpointer that has no thread, and a checkpoint id that no thread has.
    Act: run the thread with an input, or update its state, with that checkpoint id in the config.
    Assert: the call raises LookupError, as it does for a thread that has checkpoints, and the thread has no state.
    """
    builder = StateGraph(State)
    builder.add_node("write", build_writer("write"))
    builder.add_edge(Endpoint.start, "write")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(LookupError):
        call(graph, {"thread_id": "thread", "checkpoint_id": uuid4()})

    with pytest.raises(LookupError):
        graph.get_state({"thread_id": "thread"})


@pytest.mark.parametrize("finished_sibling", [False, True], ids=["failed-node-alone", "next-to-a-finished-node"])
def test_state_update_as_a_failed_node_keeps_a_deferred_node_after_the_next_step(*, finished_sibling: bool) -> None:
    """
    Arrange: a step in which a node failed, alone or next to a node that finished, an edge from the failed node to
        a writer, and a deferred node.
    Act: update the state as the failed node, and read the state.
    Assert: the next step has only the writer, because the deferred node waits until no other node runs.
    """
    builder = StateGraph(State)
    builder.add_node("failed", build_failing_once("failed", []))
    builder.add_node("writer", build_writer("writer"))
    builder.add_node("late", build_writer("late"), defer=True)
    builder.add_edge(Endpoint.start, "failed")
    builder.add_edge(Endpoint.start, "late")
    builder.add_edge("failed", "writer")
    if finished_sibling:
        builder.add_node("sibling", build_writer("sibling"))
        builder.add_edge(Endpoint.start, "sibling")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    with pytest.raises(ConnectionError):
        graph.invoke({}, config)

    graph.update_state(config, State(notes=["update"]), as_node="failed")

    actual = graph.get_state(config).next
    assert actual == ("writer",), f"Expected only the writer next, got {actual}."


def test_state_update_as_a_failed_node_keeps_the_edge_of_the_finished_node() -> None:
    """
    Arrange: a step of a node that finished and a node that failed, each with an edge to its own writer.
    Act: update the state as the failed node, and read the state.
    Assert: both writers are next, because the update keeps the writes of the finished node, its edge included.
    """
    builder = StateGraph(State)
    builder.add_node("finished", build_writer("finished"))
    builder.add_node("failed", build_failing_once("failed", []))
    builder.add_node("after finished", build_writer("after finished"))
    builder.add_node("after failed", build_writer("after failed"))
    builder.add_edge(Endpoint.start, "finished")
    builder.add_edge(Endpoint.start, "failed")
    builder.add_edge("finished", "after finished")
    builder.add_edge("failed", "after failed")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    with pytest.raises(ConnectionError):
        graph.invoke({}, config)

    graph.update_state(config, State(notes=["update"]), as_node="failed")

    state = graph.get_state(config)
    summary = (state.values, state.next)
    expected = ({"notes": ["finished", "update"]}, ("after failed", "after finished"))
    assert summary == expected, f"Expected {expected}, got {summary}."

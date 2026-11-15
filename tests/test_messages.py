import asyncio
from collections.abc import Callable
from typing import Annotated, Literal, TypedDict

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from pregel_redux import (
    Command,
    Depends,
    Endpoint,
    Entrance,
    Finished,
    InMemoryCheckpointer,
    Message,
    PersistentGraph,
    Received,
    Reducer,
    Send,
    StateGraph,
    Stopped,
    StopReason,
    Waiting,
)
from tests.conftest import join_lists, read_error


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]
    skip: bool


class Output(TypedDict):
    notes: list[str]


class Decision(BaseModel):
    approved: bool


def build_receiver(name: str, calls: list[str]) -> Callable[..., object]:
    def receive(state: State, answer: Received[str]) -> State:
        calls.append(name)
        return {"notes": [f"{name} got {answer}"]}

    return receive


def build_async_receiver(name: str, calls: list[str]) -> Callable[..., object]:
    async def receive(state: State, answer: Received[str]) -> State:
        calls.append(name)
        return {"notes": [f"{name} got {answer}"]}

    return receive


def build_writer(name: str, calls: list[str]) -> Callable[[State], State]:
    def write(state: State) -> State:
        calls.append(name)
        return {"notes": [name]}

    return write


def build_waiting_graph(
    receivers: int,
    calls: list[str],
    build_node: Callable[[str, list[str]], Callable[..., object]] = build_receiver,
    interrupt_before: tuple[str, ...] = (),
) -> PersistentGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("sibling", build_writer("sibling", calls))
    builder.add_node("after", build_writer("after", calls))
    builder.add_edge(Endpoint.start, "sibling")
    for index in range(receivers):
        builder.add_node(f"ask {index}", build_node(f"ask {index}", calls))
        builder.add_edge(Endpoint.start, f"ask {index}")
        builder.add_edge(f"ask {index}", "after")
    return builder.compile(checkpointer=InMemoryCheckpointer(), interrupt_before=interrupt_before)


def read_entrances(result: Finished[Output] | Finished[State] | Waiting | Stopped) -> list[Entrance]:
    match result:
        case Waiting():
            return list(result.accepts)
        case Finished() | Stopped():
            return []


def build_child_that_waits(calls: list[str], depth: int) -> StateGraph[State, State, State]:
    inner = StateGraph(State)
    inner.add_node("before", build_writer("before", calls))
    inner.add_node("ask", build_receiver("ask", calls))
    inner.add_edge(Endpoint.start, "before")
    inner.add_edge("before", "ask")
    for _ in range(depth - 1):
        outer = StateGraph(State)
        outer.add_node("child", inner)
        outer.add_edge(Endpoint.start, "child")
        inner = outer
    return inner


@pytest.mark.parametrize("build_node", [build_receiver, build_async_receiver], ids=["sync-node", "async-node"])
def test_run_ends_as_waiting_with_the_entrance_and_the_type_of_the_node_that_receives(
    build_node: Callable[[str, list[str]], Callable[..., object]],
) -> None:
    """
    Arrange: a graph in which one node receives a message of type str, next to a node that receives none.
    Act: run the graph on a thread.
    Assert: the result waits with the entrance of the node and its type, and only the other node ran.
    """
    calls: list[str] = []
    graph = build_waiting_graph(1, calls, build_node)

    actual = asyncio.run(graph.ainvoke({}, "thread"))

    expected = Waiting({"notes": ["sibling"]}, {Entrance((), "ask 0"): str})
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == ["sibling"], f"Expected only the call of the sibling, got {calls}."


@pytest.mark.parametrize("build_node", [build_receiver, build_async_receiver], ids=["sync-node", "async-node"])
def test_message_runs_the_node_that_waits_and_no_node_that_finished(
    build_node: Callable[[str, list[str]], Callable[..., object]],
) -> None:
    """
    Arrange: a thread that waits at one node, after another node finished.
    Act: run the thread with a Message.
    Assert: the node receives the value, and the node that finished does not run again.
    """
    calls: list[str] = []
    graph = build_waiting_graph(1, calls, build_node)
    asyncio.run(graph.ainvoke({}, "thread"))
    calls.clear()

    actual = asyncio.run(graph.ainvoke(Message("yes"), "thread"))

    expected = Finished({"notes": ["sibling", "ask 0 got yes", "after"]})
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == ["ask 0", "after"], f"Expected the calls ['ask 0', 'after'], got {calls}."


def test_state_of_a_thread_that_waits_has_no_next_node_and_the_entrance() -> None:
    """
    Arrange: a thread that waits at one node.
    Act: read the state of the thread.
    Assert: the state has no next node, and it accepts a message at the node.
    """
    graph = build_waiting_graph(1, [])
    graph.invoke({}, "thread")

    actual = graph.get_state("thread")

    expected = ((), {Entrance((), "ask 0"): str})
    assert (actual.next, actual.accepts) == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize("receivers", [2, 3])
def test_message_to_one_entrance_leaves_the_other_entrances_waiting(receivers: int) -> None:
    """
    Arrange: a thread that waits at several nodes.
    Act: run the thread with one Message for each entrance, in the order of the entrances.
    Assert: each run but the last waits at the entrances without a message, and each node gets its value.
    """
    graph = build_waiting_graph(receivers, [])
    entrances = read_entrances(graph.invoke({}, "thread"))

    results = [graph.invoke(Message(f"answer {index}", entrance), "thread") for index, entrance in enumerate(entrances)]

    waiting = [read_entrances(result) for result in results]
    expected_waiting = [entrances[index + 1 :] for index in range(receivers)]
    notes = TypeAdapter(list[str]).validate_python(graph.get_state("thread").values["notes"])
    expected_notes = {f"ask {index} got answer {index}" for index in range(receivers)}
    assert waiting == expected_waiting, f"Expected the entrances {expected_waiting}, got {waiting}."
    assert expected_notes <= set(notes), f"Expected the notes {expected_notes} among {notes}."


def test_message_without_an_entrance_for_a_thread_that_waits_at_two_nodes_is_rejected() -> None:
    """
    Arrange: a thread that waits at two nodes.
    Act: run the thread with a Message that names no entrance.
    Assert: the run raises a ValueError, and the state of the thread does not change.
    """
    graph = build_waiting_graph(2, [])
    graph.invoke({}, "thread")
    before = graph.get_state("thread")

    with pytest.raises(ValueError):
        graph.invoke(Message("answer"), "thread")

    after = graph.get_state("thread")
    assert after == before, f"Expected the state {before}, got {after}."


@pytest.mark.parametrize(
    "entrance",
    [Entrance((), "sibling"), Entrance((), "unknown")],
    ids=["node-that-receives-no-message", "unknown-node"],
)
def test_message_to_an_entrance_that_is_not_open_is_rejected(entrance: Entrance) -> None:
    """
    Arrange: a thread that waits at one node.
    Act: run the thread with a Message to another entrance.
    Assert: the run raises a ValueError, and the state of the thread does not change.
    """
    graph = build_waiting_graph(1, [])
    graph.invoke({}, "thread")
    before = graph.get_state("thread")

    with pytest.raises(ValueError):
        graph.invoke(Message("answer", entrance), "thread")

    after = graph.get_state("thread")
    assert after == before, f"Expected the state {before}, got {after}."


def test_message_to_a_thread_that_finished_is_rejected() -> None:
    """
    Arrange: a thread that finished.
    Act: run the thread with a Message.
    Assert: the run raises a ValueError.
    """
    graph = build_waiting_graph(0, [])
    graph.invoke({}, "thread")

    with pytest.raises(ValueError):
        graph.invoke(Message("answer"), "thread")


def test_message_to_a_thread_without_a_checkpoint_is_rejected() -> None:
    """
    Arrange: a graph with a checkpointer and no thread.
    Act: run a thread with a Message.
    Assert: the run raises a LookupError.
    """
    graph = build_waiting_graph(1, [])

    with pytest.raises(LookupError):
        graph.invoke(Message("answer"), "thread")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [(Decision(approved=True), "approved"), ({"approved": True}, "approved"), ({"approved": "no"}, "rejected")],
    ids=["instance-of-the-type", "mapping-of-the-fields", "mapping-with-text-for-a-bool"],
)
def test_node_receives_the_message_as_its_declared_type(answer: object, expected: str) -> None:
    """
    Arrange: a thread that waits at a node that receives a Pydantic model.
    Act: run the thread with a Message that holds an instance or a mapping of the fields.
    Assert: the node receives an instance of the model.
    """

    def decide(state: State, decision: Received[Decision]) -> State:
        return {"notes": ["approved" if decision.approved else "rejected"]}

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("decide", decide)
    builder.add_edge(Endpoint.start, "decide")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, "thread")

    actual = graph.invoke(Message(answer), "thread")

    assert actual == Finished({"notes": [expected]}), f"Expected the note {expected!r}, got {actual}."


def test_message_that_is_not_of_the_type_of_the_entrance_is_rejected() -> None:
    """
    Arrange: a thread that waits at a node that receives a Pydantic model.
    Act: run the thread with a Message that holds a text.
    Assert: the run raises the ValidationError of Pydantic, and the state of the thread does not change.
    """

    def decide(state: State, decision: Received[Decision]) -> State:
        return {}

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("decide", decide)
    builder.add_edge(Endpoint.start, "decide")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, "thread")
    before = graph.get_state("thread")

    with pytest.raises(ValidationError):
        graph.invoke(Message("text"), "thread")

    after = graph.get_state("thread")
    assert after == before, f"Expected the state {before}, got {after}."


def test_run_without_an_input_gives_the_message_again_to_a_node_that_failed() -> None:
    """
    Arrange: a thread that received a Message, in a run in which the node that received it failed.
    Act: read the state, then run the thread again without an input.
    Assert: the state has the node as its next node with the error, and the node receives the same value.
    """
    failures: list[Exception] = [ConnectionError("no network")]

    def receive_then_fail(state: State, answer: Received[str]) -> State:
        if failures:
            raise failures.pop()
        return {"notes": [answer]}

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("ask", receive_then_fail)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, "thread")
    with pytest.raises(ConnectionError):
        graph.invoke(Message("stored answer"), "thread")

    state = graph.get_state("thread")
    actual = graph.invoke(None, "thread")

    failed = [(task.name, read_error(task)) for task in state.tasks]
    assert failed == [("ask", "ConnectionError('no network')")], f"Expected the failed task of ask, got {failed}."
    assert state.accepts == {}, f"Expected no open entrance after the message, got {state.accepts}."
    assert actual == Finished({"notes": ["stored answer"]}), f"Expected the stored answer, got {actual}."


@pytest.mark.parametrize("depth", [1, 2])
def test_child_graph_that_waits_gives_its_entrance_and_continues_inside_the_child_graph(depth: int) -> None:
    """
    Arrange: a node that receives a message, in a child graph at a depth, after a node of the same child graph.
    Act: run the thread, then run it with a Message.
    Assert: the first run waits at an entrance with one segment for each level, and the second run runs only the node.
    """
    calls: list[str] = []
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("child", build_child_that_waits(calls, depth))
    builder.add_edge(Endpoint.start, "child")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    waiting = graph.invoke({}, "thread")
    state = graph.get_state("thread")
    calls.clear()
    actual = graph.invoke(Message("yes"), "thread")

    entrances = [
        ([segment.node for segment in entrance.checkpoint_ns], entrance.node) for entrance in read_entrances(waiting)
    ]
    assert entrances == [(["child"] * depth, "ask")], f"Expected the entrance of the node in the child, got {waiting}."
    assert state.next == ("child",), f"Expected the child graph as the next node of its parent, got {state.next}."
    assert calls == ["ask"], f"Expected the calls ['ask'], got {calls}."
    assert read_entrances(actual) == [], f"Expected a finished run, got {actual}."


def test_parent_graph_offers_its_own_entrance_only_after_its_step_finished() -> None:
    """
    Arrange: a thread whose root graph has a node that waits, and whose current step has a child graph that waits.
    Act: send a Message to the entrance of the root graph, then one to the child graph.
    Assert: the root entrance is refused while the step is not finished, and it is open after the child finished.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("ask", build_receiver("root ask", []))
    builder.add_node("first", build_writer("first", []))
    builder.add_node("child", build_child_that_waits([], 1))
    builder.add_edge(Endpoint.start, "ask")
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "child")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    root = Entrance((), "ask")
    during = read_entrances(graph.invoke({}, "thread"))

    with pytest.raises(ValueError):
        graph.invoke(Message("early", root), "thread")
    after = read_entrances(graph.invoke(Message("yes"), "thread"))

    assert [entrance.node for entrance in during] == ["ask"] and root not in during, (
        f"Expected only the entrance of the child graph during the step, got {during}."
    )
    assert after == [root], f"Expected the entrance of the root graph after the step, got {after}."


def test_node_that_routes_to_itself_waits_again_after_each_message() -> None:
    """
    Arrange: a graph whose node receives a message and has an edge to itself.
    Act: run the thread, then run it with three messages.
    Assert: each run waits at the same entrance, and the state has one note for each message.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("listen", build_receiver("listen", []))
    builder.add_edge(Endpoint.start, "listen")
    builder.add_edge("listen", "listen")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, "thread")

    results = [graph.invoke(Message(f"message {index}"), "thread") for index in range(3)]

    expected = Waiting({"notes": [f"listen got message {index}" for index in range(3)]}, {Entrance((), "listen"): str})
    assert results[-1] == expected, f"Expected {expected}, got {results[-1]}."


def test_deferred_node_runs_after_the_node_that_waits_received_its_message() -> None:
    """
    Arrange: a graph with a node that receives a message and a deferred node, both after the start.
    Act: run the thread, then run it with a Message.
    Assert: the deferred node does not run while the other node waits, and it runs after that node.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("ask", build_receiver("ask", []))
    builder.add_node("late", build_writer("late", []), defer=True)
    builder.add_edge(Endpoint.start, "ask")
    builder.add_edge(Endpoint.start, "late")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    waiting = graph.invoke({}, "thread")
    actual = graph.invoke(Message("yes"), "thread")

    assert waiting == Waiting({"notes": []}, {Entrance((), "ask"): str}), f"Expected no note yet, got {waiting}."
    expected = Finished({"notes": ["ask got yes", "late"]})
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_join_starts_a_node_that_receives_a_message_only_after_the_message() -> None:
    """
    Arrange: a graph in which a join of two nodes leads to a node that receives a message.
    Act: run the thread, then run it with a Message.
    Assert: the first run waits at the node after the two sources ran, and the second run runs the node one time.
    """
    calls: list[str] = []
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("left", build_writer("left", calls))
    builder.add_node("right", build_writer("right", calls))
    builder.add_node("ask", build_receiver("ask", calls))
    builder.add_edge(Endpoint.start, "left")
    builder.add_edge(Endpoint.start, "right")
    builder.add_edge(("left", "right"), "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    waiting = graph.invoke({}, "thread")
    actual = graph.invoke(Message("yes"), "thread")

    assert read_entrances(waiting) == [Entrance((), "ask")], (
        f"Expected a wait at the target of the join, got {waiting}."
    )
    assert actual == Finished({"notes": ["left", "right", "ask got yes"]}), f"Expected one run of ask, got {actual}."
    assert (sorted(calls[:2]), calls[2:]) == (["left", "right"], ["ask"]), (
        f"Expected the calls of the two sources and then one call of ask, got {calls}."
    )


def test_new_input_discards_the_wait_of_a_node() -> None:
    """
    Arrange: a thread that waits at a node, in a graph whose first node routes to that node only without "skip".
    Act: run the thread with a new input that has "skip".
    Assert: the run finishes, because the new input starts the flow again and the node does not wait any more.
    """

    def route(state: State) -> Command[Literal["ask"]]:
        return Command(goto=[] if state.get("skip") else "ask")

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("route", route)
    builder.add_node("ask", build_receiver("ask", []))
    builder.add_edge(Endpoint.start, "route")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    waiting = graph.invoke({}, "thread")

    actual = graph.invoke({"skip": True}, "thread")

    assert read_entrances(waiting) == [Entrance((), "ask")], f"Expected a wait in the first run, got {waiting}."
    assert actual == Finished(Output(notes=[])), f"Expected a finished run, got {actual}."


def test_breakpoint_before_a_node_that_received_its_message_stops_the_run() -> None:
    """
    Arrange: a graph with a static stop before the node that receives a message, and a thread that waits at it.
    Act: run the thread with a Message, then run it without an input.
    Assert: the first run stops before the node, and the second run gives the value to the node.
    """
    calls: list[str] = []
    graph = build_waiting_graph(1, calls, interrupt_before=("ask 0",))
    graph.invoke({}, "thread")
    calls.clear()

    stopped = graph.invoke(Message("yes"), "thread")
    stopped_calls = list(calls)
    actual = graph.invoke(None, "thread")

    assert stopped == Stopped({"notes": ["sibling"]}, StopReason.breakpoint), f"Expected a stop, got {stopped}."
    assert stopped_calls == [], f"Expected no call before the stop, got {stopped_calls}."
    expected = Finished({"notes": ["sibling", "ask 0 got yes", "after"]})
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_fork_from_the_checkpoint_that_waits_takes_another_message() -> None:
    """
    Arrange: a thread that waited at a node and then finished with one message.
    Act: run the thread with another Message, from the checkpoint that waited.
    Assert: the run finishes with the note of the other message and without the note of the first message.
    """
    graph = build_waiting_graph(1, [])
    graph.invoke({}, "thread")
    waited = graph.get_state("thread").checkpoint_id
    graph.invoke(Message("first"), "thread")

    actual = graph.invoke(Message("second"), "thread", fork_from=waited)

    expected = Finished({"notes": ["sibling", "ask 0 got second", "after"]})
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_fork_from_a_checkpoint_with_a_next_node_takes_no_message() -> None:
    """
    Arrange: a thread that waits at a node, with an older checkpoint that has a next node.
    Act: run the thread with a Message, from the older checkpoint.
    Assert: the run raises a ValueError, and the thread gets no new checkpoint.
    """
    graph = build_waiting_graph(1, [])
    graph.invoke({}, "thread")
    before = list(graph.get_state_history("thread"))
    older = next(state for state in before if state.next)

    with pytest.raises(ValueError):
        graph.invoke(Message("answer"), "thread", fork_from=older.checkpoint_id)

    after = list(graph.get_state_history("thread"))
    assert after == before, f"Expected the history {before}, got {after}."


@pytest.mark.parametrize(
    ("command", "note", "count"),
    [(Command[str](update={"notes": ["edit"]}), "edit", 1), (Command[str](goto="after"), "after", 2)],
    ids=["update", "goto"],
)
def test_run_with_a_command_keeps_the_wait_of_a_node(command: Command[str], note: str, count: int) -> None:
    """
    Arrange: a thread that waits at a node that has an edge to the node "after".
    Act: run the thread with a Command that has an update or a goto, then run it with a Message.
    Assert: the node still waits after the Command, and the state has the note of the Command.
    """
    graph = build_waiting_graph(1, [])
    graph.invoke({}, "thread")

    commanded = graph.invoke(command, "thread")
    graph.invoke(Message("yes"), "thread")

    notes = TypeAdapter(list[str]).validate_python(graph.get_state("thread").values["notes"])
    assert read_entrances(commanded) == [Entrance((), "ask 0")], (
        f"Expected the wait after the Command, got {commanded}."
    )
    assert notes.count(note) == count, f"Expected the note {note!r} {count} time(s), got {notes}."


@pytest.mark.parametrize("interrupt_before", [[], ["late"]], ids=["finished-thread", "thread-before-the-node"])
def test_goto_of_a_command_runs_a_deferred_node(interrupt_before: list[str]) -> None:
    """
    Arrange: a thread of a graph with two deferred nodes, which finished or which stopped before the nodes.
    Act: run the thread with a Command whose goto names the deferred node.
    Assert: a finished thread runs the node one more time, and a stopped thread runs the two nodes in one step.
    """
    builder = StateGraph(State)
    builder.add_node("late", build_writer("late", []), defer=True)
    builder.add_node("other", build_writer("other", []), defer=True)
    builder.add_edge(Endpoint.start, "late")
    builder.add_edge(Endpoint.start, "other")
    graph = builder.compile(checkpointer=InMemoryCheckpointer(), interrupt_before=interrupt_before)
    asyncio.run(graph.ainvoke({}, "thread"))
    before = asyncio.run(graph.aget_state("thread")).values.get("notes", [])

    actual = asyncio.run(graph.ainvoke(Command(goto="late"), "thread"))

    expected = Finished(State(notes=["late", "other", "late"] if before else ["late", "other"]))
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_send_to_a_node_that_receives_a_message_fails_the_node_that_sends() -> None:
    """
    Arrange: a graph whose first node returns a Send to a node that receives a message.
    Act: run the thread.
    Assert: the run raises a ValueError, because a Send cannot wait for a message.
    """

    def send(state: State) -> Command[Literal["ask"]]:
        return Command(goto=Send("ask", State(notes=["payload"])))

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("send", send)
    builder.add_node("ask", build_receiver("ask", []))
    builder.add_edge(Endpoint.start, "send")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(ValueError):
        graph.invoke({}, "thread")


def test_dependency_of_a_node_receives_the_message_of_the_node() -> None:
    """
    Arrange: a node that receives a message, with a dependency that requests the received message.
    Act: run the thread, then run it with a Message.
    Assert: the dependency gets the value of the Message.
    """

    def count_letters(message: Received[str]) -> int:
        return len(message)

    def ask(state: State, answer: Received[str], letters: Annotated[int, Depends(count_letters)]) -> State:
        return {"notes": [f"{answer} has {letters} letters"]}

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, "thread")

    actual = graph.invoke(Message("yes"), "thread")

    assert actual == Finished({"notes": ["yes has 3 letters"]}), f"Expected the count of the letters, got {actual}."


def test_dependency_that_requests_a_message_of_a_node_that_receives_none_is_rejected() -> None:
    """
    Arrange: a node that receives no message, with a dependency that requests the received message.
    Act: compile the graph.
    Assert: the compile raises a TypeError.
    """

    def count_letters(message: Received[str]) -> int:
        return len(message)

    def write(state: State, letters: Annotated[int, Depends(count_letters)]) -> State:
        return {}

    builder = StateGraph(State)
    builder.add_node("write", write)
    builder.add_edge(Endpoint.start, "write")

    with pytest.raises(TypeError):
        builder.compile(checkpointer=InMemoryCheckpointer())


def test_node_that_receives_its_message_as_two_types_is_rejected() -> None:
    """
    Arrange: a node with two parameters that receive the message, as two different types.
    Act: compile the graph.
    Assert: the compile raises a TypeError.
    """

    def ask(state: State, text: Received[str], number: Received[int]) -> State:
        return {}

    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "ask")

    with pytest.raises(TypeError):
        builder.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_graph_without_a_checkpointer_that_has_a_node_that_receives_is_rejected(depth: int) -> None:
    """
    Arrange: a graph in which a node receives a message, in the graph or in a child graph at a depth.
    Act: compile the graph without a checkpointer.
    Assert: the compile raises a ValueError, because only a thread can wait for a message.
    """
    builder = StateGraph(State)
    if depth == 0:
        builder.add_node("ask", build_receiver("ask", []))
        builder.add_edge(Endpoint.start, "ask")
    else:
        builder.add_node("child", build_child_that_waits([], depth))
        builder.add_edge(Endpoint.start, "child")

    with pytest.raises(ValueError):
        builder.compile()


def test_deferred_node_that_receives_a_message_is_rejected() -> None:
    """
    Arrange: a graph with a deferred node that receives a message.
    Act: compile the graph.
    Assert: the compile raises a ValueError.
    """
    builder = StateGraph(State)
    builder.add_node("ask", build_receiver("ask", []), defer=True)
    builder.add_edge(Endpoint.start, "ask")

    with pytest.raises(ValueError):
        builder.compile(checkpointer=InMemoryCheckpointer())


def test_checkpoint_with_a_message_for_a_node_that_receives_none_in_another_graph_is_rejected() -> None:
    """
    Arrange: a thread with a delivered message whose node failed, and a graph in which that node receives no message.
    Act: run the thread on the other graph.
    Assert: the run raises a ValueError, because the checkpoint does not fit the graph.
    """

    def fail(state: State, answer: Received[str]) -> State:
        raise ConnectionError

    checkpointer = InMemoryCheckpointer()
    builder = StateGraph(State)
    builder.add_node("ask", fail)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({}, "thread")
    with pytest.raises(ConnectionError):
        graph.invoke(Message("yes"), "thread")
    other = StateGraph(State)
    other.add_node("ask", build_writer("ask", []))
    other.add_edge(Endpoint.start, "ask")

    with pytest.raises(ValueError):
        other.compile(checkpointer=checkpointer).invoke(None, "thread")

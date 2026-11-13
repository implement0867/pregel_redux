import asyncio
from collections.abc import Callable
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest

from pregel_redux import (
    Command,
    Depends,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupted,
    PersistentGraph,
    Reducer,
    Scope,
    Send,
    StateGraph,
    StateSnapshot,
    StreamMode,
    ThreadConfig,
    interrupt,
    read_config,
)
from pregel_redux.checkpoints import CheckpointSource
from pregel_redux.snapshots import PullPath, PushPath
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
    graph.invoke({"notes": ["input"]}, {"thread_id": "thread"})

    actual = graph.get_state({"thread_id": "thread"})
    from_async = asyncio.run(graph.aget_state({"thread_id": "thread"}))

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
    graph.invoke({}, {"thread_id": "thread"})

    actual = graph.get_state({"thread_id": "thread"})

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

    with pytest.raises(LookupError):
        graph.get_state({"thread_id": "missing"})


@pytest.mark.parametrize(("skipped", "limit"), [(0, None), (2, None), (0, 2), (1, 1)])
def test_state_history_gives_the_states_before_an_id_with_the_newest_first(skipped: int, limit: int | None) -> None:
    """
    Arrange: a thread of a run that finished.
    Act: read the history of the thread before the id of one state, with a limit.
    Assert: the history has the older states with the newest first, and each state names the node that runs next.
    """
    graph = build_chain([])
    graph.invoke({}, {"thread_id": "thread"})
    history = list(graph.get_state_history({"thread_id": "thread"}))
    before = None if skipped == 0 else history[skipped - 1].checkpoint_id

    actual = asyncio.run(collect(graph.aget_state_history({"thread_id": "thread"}, before=before, limit=limit)))

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
    expected = graph.invoke({}, {"thread_id": "thread"})
    earlier = next(state for state in graph.get_state_history({"thread_id": "thread"}) if state.next == ("second",))
    calls.clear()

    actual = graph.invoke(None, {"thread_id": "thread", "checkpoint_id": earlier.checkpoint_id})

    history = list(graph.get_state_history({"thread_id": "thread"}))
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
    graph.invoke({}, {"thread_id": "thread"})
    graph.invoke({}, {"thread_id": "other"})
    other_id = graph.get_state({"thread_id": "other"}).checkpoint_id

    with pytest.raises(LookupError):
        graph.invoke(None, {"thread_id": "thread", "checkpoint_id": other_id})


@pytest.mark.parametrize("as_node", ["first", None])
def test_state_update_counts_as_the_output_of_a_node(as_node: str | None) -> None:
    """
    Arrange: a thread that stopped before the second node.
    Act: update the state as the first node, by name or by inference, then continue the run.
    Assert: the new state has the update and the second node next, and the run finishes with the update.
    """
    calls: list[str] = []
    graph = build_chain(calls, interrupt_before=("second",))
    graph.invoke({}, {"thread_id": "thread"})

    checkpoint_id = graph.update_state({"thread_id": "thread"}, {"notes": ["edit"]}, as_node=as_node)

    state = graph.get_state({"thread_id": "thread"})
    summary = (state.checkpoint_id, state.values, state.next, state.source)
    expected = (checkpoint_id, {"notes": ["first", "edit"]}, ("second",), CheckpointSource.update)
    actual = graph.invoke(None, {"thread_id": "thread"})
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
    graph.invoke({}, {"thread_id": "thread"})

    asyncio.run(graph.aupdate_state({"thread_id": "thread"}, {"notes": ["edit"]}, as_node="second"))

    actual = graph.invoke(None, {"thread_id": "thread"})
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

    graph.update_state({"thread_id": "thread"}, {"notes": ["edit"]}, as_node="first")

    actual = graph.invoke(None, {"thread_id": "thread"})
    assert actual == Finished(Output(notes=["edit", "second", "third"])), f"Got {actual}."
    assert calls == ["second", "third"], f"Expected the calls ['second', 'third'], got {calls}."


@pytest.mark.parametrize(
    ("as_node", "started"),
    [(None, False), ("missing", True), ("missing", False)],
)
def test_state_update_without_a_clear_node_is_rejected(as_node: str | None, *, started: bool) -> None:
    """
    Arrange: a graph with three nodes, and a thread with or without checkpoints.
    Act: update the state with no node that the graph can infer, or as a node that the graph does not have.
    Assert: the call raises ValueError.
    """
    graph = build_chain([])
    if started:
        graph.invoke({}, {"thread_id": "thread"})

    with pytest.raises(ValueError):
        graph.update_state({"thread_id": "thread"}, {"notes": ["edit"]}, as_node=as_node)


def test_state_update_infers_a_node_that_ran_after_an_edge_and_a_join() -> None:
    """
    Arrange: a thread in which the last node ran after an edge and a join that arrived in the same step.
    Act: update the state with no node name.
    Assert: the update counts as the output of the last node, so the node after it is next.
    """
    builder = StateGraph(State, output_schema=Output)
    for name in ("left", "right", "both", "last"):
        builder.add_node(name, build_writer(name, []))
    builder.add_edge(Endpoint.start, "left")
    builder.add_edge(Endpoint.start, "right")
    builder.add_edge("left", "both")
    builder.add_edge(("left", "right"), "both")
    builder.add_edge("both", "last")
    graph = builder.compile(checkpointer=InMemoryCheckpointer(), interrupt_after=("both",))
    graph.invoke({}, {"thread_id": "thread"})

    graph.update_state({"thread_id": "thread"}, {"notes": ["edit"]})

    state = graph.get_state({"thread_id": "thread"})
    summary = (state.values, state.next)
    expected = ({"notes": ["left", "right", "both", "edit"]}, ("last",))
    assert summary == expected, f"Expected {expected}, got {summary}."


def test_state_update_from_an_earlier_checkpoint_becomes_the_newest_state() -> None:
    """
    Arrange: a thread of a run that finished, and the id of the state before the third node.
    Act: update the state from that checkpoint as the second node, then continue the run.
    Assert: the run continues from the update, so only the third node runs.
    """
    calls: list[str] = []
    graph = build_chain(calls)
    graph.invoke({}, {"thread_id": "thread"})
    earlier = next(state for state in graph.get_state_history({"thread_id": "thread"}) if state.next == ("third",))
    calls.clear()

    graph.update_state({"thread_id": "thread", "checkpoint_id": earlier.checkpoint_id}, {"notes": ["edit"]})

    actual = graph.invoke(None, {"thread_id": "thread"})
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

    result = graph.invoke({}, {"thread_id": "thread"})
    while isinstance(result, Interrupted):
        assert result.interrupts == (), f"Expected a stop without an interrupt value, got {result}."
        stops.append([*calls])
        result = graph.invoke(None, {"thread_id": "thread"})

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
    graph.invoke({}, {"thread_id": "thread"})

    actual = graph.invoke({"notes": ["more"]}, {"thread_id": "thread"})

    assert actual == Interrupted((), {"notes": ["first", "more", "first"]}), f"Expected a stop, got {actual}."
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

    stopped = [graph.invoke({}, {"thread_id": "thread"})]
    actual = graph.invoke(None, {"thread_id": "thread"})

    assert stopped == [Interrupted((), {"notes": []})], f"Expected one stop, got {stopped}."
    assert calls == ["first", "second"], f"Expected each node to run one time, got {calls}."
    assert isinstance(actual, Finished), f"Expected a finished run, got {actual}."


@pytest.mark.parametrize(
    ("interrupt_before", "interrupt_after"), [(["missing"], []), ([], ["missing"])], ids=["before", "after"]
)
def test_unknown_node_in_an_interrupt_list_is_rejected(interrupt_before: list[str], interrupt_after: list[str]) -> None:
    """
    Arrange: a builder with one node.
    Act: compile it with interrupt_before or interrupt_after that names another node.
    Assert: compile raises ValueError that names the node.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("first", build_writer("first", []))
    builder.add_edge(Endpoint.start, "first")

    with pytest.raises(ValueError, match="missing"):
        builder.compile(
            checkpointer=InMemoryCheckpointer(), interrupt_before=interrupt_before, interrupt_after=interrupt_after
        )


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

    first = graph.invoke({}, {"thread_id": "thread"})
    second = graph.invoke(None, {"thread_id": "thread"})
    third = graph.invoke(Command(resume="answer"), {"thread_id": "thread"})

    values = [item.value for item in second.interrupts] if isinstance(second, Interrupted) else None
    assert first == Interrupted((), {"notes": []}), f"Expected a stop without a value, got {first}."
    assert values == ["question"], f"Expected the value of the interrupt, got {second}."
    assert third == Finished(Output(notes=["answer"])), f"Expected the answer, got {third}."


def build_parent_of_asking_child(depth: int) -> PersistentGraph[State, State, Output]:
    graph = StateGraph(State, output_schema=State)
    graph.add_node("first", build_writer("first", []))
    graph.add_node("ask", ask)
    graph.add_edge(Endpoint.start, "first")
    graph.add_edge("first", "ask")
    for _ in range(depth - 1):
        parent = StateGraph(State, output_schema=State)
        parent.add_node("child", graph)
        parent.add_edge(Endpoint.start, "child")
        graph = parent
    root = StateGraph(State, output_schema=Output)
    root.add_node("child", graph)
    root.add_edge(Endpoint.start, "child")
    return root.compile(checkpointer=InMemoryCheckpointer())


def read_deepest_state(snapshot: StateSnapshot) -> StateSnapshot:
    children = [task.state for task in snapshot.tasks if task.state is not None]
    return read_deepest_state(children[0]) if children else snapshot


@pytest.mark.parametrize("depth", [1, 2])
def test_state_with_subgraphs_has_the_state_of_a_child_graph_that_stopped(depth: int) -> None:
    """
    Arrange: a thread that stopped at an interrupt in a child graph at a given depth.
    Act: read the state with subgraphs and without subgraphs.
    Assert: the state with subgraphs has the values, next node and interrupt of the deepest child graph.
    """
    graph = build_parent_of_asking_child(depth)
    graph.invoke({}, {"thread_id": "thread"})

    with_children = graph.get_state({"thread_id": "thread"}, subgraphs=True)
    without_children = asyncio.run(graph.aget_state({"thread_id": "thread"}))

    actual = read_deepest_state(with_children)
    values = [item.value for item in actual.interrupts]
    assert len(actual.checkpoint_ns.split("|")) == depth, f"Expected a namespace of depth {depth}, got {actual}."
    assert actual.values == {"notes": ["first"]}, f"Expected the note of the first node, got {actual.values}."
    assert actual.next == ("ask",), f"Expected the next node ask, got {actual.next}."
    assert values == ["question"], f"Expected the interrupt of the child graph, got {values}."
    assert [task.state for task in without_children.tasks] == [None], "Expected no child state without subgraphs."


def test_state_of_a_child_graph_is_read_by_its_namespace() -> None:
    """
    Arrange: a thread that stopped in a grandchild graph, and the state of the thread with subgraphs.
    Act: read the state with the namespace of the grandchild graph, and with a namespace that names no child graph.
    Assert: the state equals the state of the grandchild graph, and the unknown namespace raises LookupError.
    """
    graph = build_parent_of_asking_child(2)
    graph.invoke({}, {"thread_id": "thread"})
    expected = read_deepest_state(graph.get_state({"thread_id": "thread"}, subgraphs=True))

    actual = graph.get_state({"thread_id": "thread", "checkpoint_ns": expected.checkpoint_ns})

    assert actual == expected, f"Expected {expected}, got {actual}."
    with pytest.raises(LookupError):
        graph.get_state({"thread_id": "thread", "checkpoint_ns": "missing:1"})


def test_update_of_the_state_of_a_child_graph_is_in_the_output() -> None:
    """
    Arrange: a thread that stopped at an interrupt in a child graph.
    Act: update the state of the child graph as its first node, then resume the thread.
    Assert: the output has the note of the first node, the update and the answer.
    """
    graph = build_parent_of_asking_child(1)
    graph.invoke({}, {"thread_id": "thread"})
    child = read_deepest_state(graph.get_state({"thread_id": "thread"}, subgraphs=True))
    graph.update_state(
        {"thread_id": "thread", "checkpoint_ns": child.checkpoint_ns}, {"notes": ["update"]}, as_node="first"
    )

    actual = graph.invoke(Command(resume="answer"), {"thread_id": "thread"})

    expected = Finished(Output(notes=["first", "update", "answer"]))
    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize(
    ("stop_before", "notes"), [("second", ["input", "first"]), ("third", ["input", "first", "second"])]
)
def test_interrupted_result_has_the_values_at_the_stop(stop_before: str, notes: list[str]) -> None:
    """
    Arrange: a chain of nodes that stops before a named node.
    Act: run a thread with an input.
    Assert: the result has no interrupt and the values that the nodes before the stop wrote.
    """
    graph = build_chain([], interrupt_before=(stop_before,))
    expected = Interrupted((), {"notes": notes})

    actual = graph.invoke({"notes": ["input"]}, {"thread_id": "thread"})

    assert actual == expected, f"Expected {expected}, got {actual}."


def test_state_tasks_have_the_result_of_a_finished_task_and_the_path() -> None:
    """
    Arrange: a thread whose step has one node that finished and one node that stopped at an interrupt.
    Act: read the state of the thread.
    Assert: the finished task has its update as result, the other task has no result, and each path names the node.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("write", build_writer("write", []))
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "write")
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, {"thread_id": "thread"})
    expected = {"write": (PullPath("write"), {"notes": ["write"]}), "ask": (PullPath("ask"), None)}

    actual = graph.get_state({"thread_id": "thread"})

    summary = {task.name: (task.path, task.result) for task in actual.tasks}
    assert summary == expected, f"Expected {expected}, got {summary}."


@pytest.mark.parametrize("sends", [1, 2, 3])
def test_state_task_of_a_send_has_the_index_of_the_send_as_path(sends: int) -> None:
    """
    Arrange: a thread whose node sent a number of inputs to a node that stops at an interrupt.
    Act: read the state of the thread.
    Assert: the path of each task has the index of its send.
    """

    def fan_out(state: State) -> Command[str]:
        return Command(goto=[Send("ask", {}) for _ in range(sends)])

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("fan_out", fan_out)
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "fan_out")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, {"thread_id": "thread"})
    expected = [PushPath(index) for index in range(sends)]

    actual = graph.get_state({"thread_id": "thread"})

    paths = [task.path for task in actual.tasks]
    assert paths == expected, f"Expected the paths {expected}, got {paths}."


def read_thread(state: State, config: Annotated[ThreadConfig, Depends(read_config)]) -> State:
    return {"notes": [config["thread_id"]]}


async def read_thread_async(state: State, config: Annotated[ThreadConfig, Depends(read_config)]) -> State:
    return {"notes": [config["thread_id"]]}


def provide_thread(config: Annotated[ThreadConfig, Depends(read_config, scope=Scope.run)]) -> str:
    return config["thread_id"]


def read_provided_thread(state: State, thread: Annotated[str, Depends(provide_thread, scope=Scope.run)]) -> State:
    return {"notes": [thread]}


@pytest.mark.parametrize("node", [read_thread, read_thread_async, read_provided_thread])
@pytest.mark.parametrize("depth", [0, 1, 2], ids=["node-of-the-graph", "node-of-a-child", "node-of-a-grandchild"])
@pytest.mark.parametrize("thread_id", ["first-thread", "second-thread"])
def test_node_reads_the_thread_id_of_the_run(node: Callable[..., object], depth: int, thread_id: str) -> None:
    """
    Arrange: a graph with a checkpointer and a node, at a given depth of child graphs, that requests the config.
    Act: run a thread.
    Assert: the node wrote the thread id of the run.
    """
    graph = StateGraph(State, output_schema=State)
    graph.add_node("read", node)
    graph.add_edge(Endpoint.start, "read")
    for _ in range(depth):
        parent = StateGraph(State, output_schema=State)
        parent.add_node("child", graph)
        parent.add_edge(Endpoint.start, "child")
        graph = parent
    expected = Finished(State(notes=[thread_id]))

    actual = asyncio.run(graph.compile(checkpointer=InMemoryCheckpointer()).ainvoke({}, {"thread_id": thread_id}))

    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize("node", [read_thread, read_thread_async, read_provided_thread])
def test_node_of_a_graph_without_a_checkpointer_cannot_read_the_config(node: Callable[..., object]) -> None:
    """
    Arrange: a graph without a checkpointer and a node that requests the config.
    Act: run the graph.
    Assert: the run raises RuntimeError that names the checkpointer.
    """
    builder = StateGraph(State, output_schema=State)
    builder.add_node("read", node)
    builder.add_edge(Endpoint.start, "read")

    with pytest.raises(RuntimeError):
        asyncio.run(builder.compile().ainvoke({}))


def run_with_invoke(graph: PersistentGraph[State, State, Output], config: ThreadConfig) -> object:
    return graph.invoke({}, config)


def run_with_stream(graph: PersistentGraph[State, State, Output], config: ThreadConfig) -> object:
    return list(graph.stream({}, config, stream_mode=StreamMode.values))


@pytest.mark.parametrize("run", [run_with_invoke, run_with_stream])
def test_run_with_a_namespace_in_the_config_is_rejected(
    run: Callable[[PersistentGraph[State, State, Output], ThreadConfig], object],
) -> None:
    """
    Arrange: a graph with a checkpointer and a config that has a checkpoint namespace.
    Act: run a thread with the config.
    Assert: the run raises ValueError that names checkpoint_ns, and no node ran.
    """
    calls: list[str] = []
    graph = build_chain(calls)

    with pytest.raises(ValueError):
        run(graph, {"thread_id": "thread", "checkpoint_ns": "child:1"})

    assert calls == [], f"Expected no node to run, got {calls}."


def test_state_history_of_a_child_graph_is_read_by_its_namespace() -> None:
    """
    Arrange: a thread that stopped in a child graph, and the namespace of the child graph.
    Act: read the history of the thread with the namespace.
    Assert: the history has the states of the child graph with the newest first.
    """
    graph = build_parent_of_asking_child(1)
    graph.invoke({}, {"thread_id": "thread"})
    child = read_deepest_state(graph.get_state({"thread_id": "thread"}, subgraphs=True))
    expected = [("ask",), ("first",), ("__start__",)]

    actual = list(graph.get_state_history({"thread_id": "thread", "checkpoint_ns": child.checkpoint_ns}))

    nodes = [state.next for state in actual]
    assert nodes == expected, f"Expected the next nodes {expected}, got {nodes}."
    assert actual[0] == child, f"Expected the newest state {child}, got {actual[0]}."


@pytest.mark.parametrize(
    ("before_index", "limit", "found"),
    [(None, None, True), (1, None, True), (None, 1, True), (2, None, False), (3, None, False), (None, 0, False)],
)
def test_state_history_with_a_checkpoint_id_has_only_that_state(
    before_index: int | None, limit: int | None, *, found: bool
) -> None:
    """
    Arrange: a thread of a run that finished, and the id of the third newest state.
    Act: read the history with that id in the config, before the id of another state, with a limit.
    Assert: the history has only that state, or no state when the state is not before the given id or the limit is 0.
    """
    graph = build_chain([])
    graph.invoke({}, {"thread_id": "thread"})
    history = list(graph.get_state_history({"thread_id": "thread"}))
    before = None if before_index is None else history[before_index].checkpoint_id
    expected = [history[2]] if found else []

    actual = list(
        graph.get_state_history(
            {"thread_id": "thread", "checkpoint_id": history[2].checkpoint_id}, before=before, limit=limit
        )
    )

    assert actual == expected, f"Expected {expected}, got {actual}."


def test_state_history_with_an_unknown_checkpoint_id_is_empty() -> None:
    """
    Arrange: a thread of a run that finished.
    Act: read the history with a checkpoint id that the thread does not have.
    Assert: the history is empty.
    """
    graph = build_chain([])
    graph.invoke({}, {"thread_id": "thread"})

    actual = list(graph.get_state_history({"thread_id": "thread", "checkpoint_id": uuid4()}))

    assert actual == [], f"Expected no state, got {actual}."


def test_state_update_after_a_failed_step_keeps_the_writes_of_the_finished_node() -> None:
    """
    Arrange: a thread whose step has a node that finished and a node that failed.
    Act: update the state as the failed node, and read the state.
    Assert: the state has the write of the finished node and then the update.
    """
    calls: list[str] = []

    def fail(state: State) -> State:
        raise ConnectionError

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("finished", build_writer("finished", calls))
    builder.add_node("failed", fail)
    builder.add_edge(Endpoint.start, "finished")
    builder.add_edge(Endpoint.start, "failed")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    config = ThreadConfig(thread_id="thread")
    with pytest.raises(ConnectionError):
        graph.invoke({}, config)

    graph.update_state(config, State(notes=["update"]), as_node="failed")
    actual = graph.get_state(config).values

    expected = {"notes": ["finished", "update"]}
    assert actual == expected, f"Expected {expected}, got {actual}."

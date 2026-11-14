import asyncio
from collections.abc import Callable
from typing import Annotated, TypedDict

import pytest

from pregel_redux import (
    Command,
    CompiledGraph,
    Endpoint,
    InMemoryCheckpointer,
    Reducer,
    StateGraph,
    get_stream_writer,
    interrupt,
)
from pregel_redux.streams import (
    CheckpointPart,
    ChildPart,
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


class Output(TypedDict):
    notes: list[str]


def build_writer(name: str) -> Callable[[State], State]:
    def write(state: State) -> State:
        get_stream_writer()(f"{name} works")
        return {"notes": [name]}

    return write


async def write_async(state: State) -> State:
    get_stream_writer()("async works")
    return {"notes": ["async"]}


def ask(state: State) -> State:
    return {"notes": [interrupt("question", response_schema=str)]}


def fail(state: State) -> State:
    raise ConnectionError("no network")


def build_builder(*nodes: tuple[str, Callable[[State], object]]) -> StateGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    previous: str | None = None
    for name, node in nodes:
        builder.add_node(name, node)
        builder.add_edge(Endpoint.start if previous is None else previous, name)
        previous = name
    return builder


def build_chain() -> CompiledGraph[State, State, Output]:
    return build_builder(("first", build_writer("first")), ("second", build_writer("second"))).compile()


def build_parent() -> CompiledGraph[State, State, Output]:
    child = StateGraph(State, output_schema=State)
    child.add_node("inner", build_writer("inner"))
    child.add_edge(Endpoint.start, "inner")
    parent = StateGraph(State, output_schema=Output)
    parent.add_node("child", child)
    parent.add_node("last", build_writer("last"))
    parent.add_edge(Endpoint.start, "child")
    parent.add_edge("child", "last")
    return parent.compile()


def test_stream_gives_each_part_in_the_order_of_the_run() -> None:
    """
    Arrange: a graph with one node that writes to the stream writer.
    Act: stream a run.
    Assert: the stream has the state after the input, then the start, the data, the update and the result
    of the node, then the state after the node.
    """
    graph = build_builder(("first", build_writer("first"))).compile()

    actual = asyncio.run(collect(graph.astream({})))

    ids = [part.id for part in actual if isinstance(part, TaskStartPart)]
    expected: list[StreamPart | ChildPart] = [
        ValuesPart({"notes": []}, ()),
        TaskStartPart(ids[0], "first", {"notes": []}),
        CustomPart("first works"),
        UpdatesPart("first", {"notes": ["first"]}),
        TaskResultPart(ids[0], "first", {"notes": ["first"]}, ()),
        ValuesPart({"notes": ["first"]}, ()),
    ]
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_values_parts_give_the_state_after_each_step() -> None:
    """
    Arrange: a graph of two nodes in a chain.
    Act: stream a run.
    Assert: the values parts have the state after the input and after each node.
    """
    graph = build_chain()

    parts = asyncio.run(collect(graph.astream({"notes": ["input"]})))

    actual = [part for part in parts if isinstance(part, ValuesPart)]
    expected = [
        ValuesPart({"notes": ["input"]}, ()),
        ValuesPart({"notes": ["input", "first"]}, ()),
        ValuesPart({"notes": ["input", "first", "second"]}, ()),
    ]
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_custom_parts_give_the_data_that_a_node_writes() -> None:
    """
    Arrange: a graph with a sync node and an async node that write to the stream writer.
    Act: stream a run.
    Assert: the custom parts have the data of each node in the order of the nodes.
    """
    graph = build_builder(("first", build_writer("first")), ("async", write_async)).compile()

    parts = asyncio.run(collect(graph.astream({})))

    actual = [part for part in parts if isinstance(part, CustomPart)]
    expected = [CustomPart("first works"), CustomPart("async works")]
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_parts_of_a_child_graph_are_child_parts_with_the_namespace_of_the_child_graph() -> None:
    """
    Arrange: a parent graph with a child graph that has one node.
    Act: stream a run.
    Assert: the update of the inner node is in a child part with the namespace of the child graph, and the
    updates of the parent graph are not in child parts.
    """
    graph = build_parent()

    parts = asyncio.run(collect(graph.astream({})))

    outer = [part.node for part in parts if isinstance(part, UpdatesPart)]
    inner = [
        (tuple(segment.split(":")[0] for segment in part.ns), part.part.node)
        for part in parts
        if isinstance(part, ChildPart) and isinstance(part.part, UpdatesPart)
    ]
    assert outer == ["child", "last"], f"Expected the updates of the parent graph, got {outer}."
    assert inner == [(("child",), "inner")], f"Expected the update of the inner node, got {inner}."


def test_sync_stream_gives_the_parts_of_the_async_stream() -> None:
    """
    Arrange: a graph of two sync nodes in a chain.
    Act: stream a run with the sync function, outside and inside a running event loop.
    Assert: each stream is equal to the async stream.
    """
    graph = build_chain()
    expected = asyncio.run(collect(graph.astream({})))

    async def stream_inside_a_loop() -> list[StreamPart | ChildPart]:
        return list(graph.stream({}))

    actual = list(graph.stream({}))
    inside = asyncio.run(stream_inside_a_loop())

    assert actual == expected, f"Expected {expected}, got {actual}."
    assert inside == expected, f"Expected {expected}, got {inside}."


def test_sync_stream_rejects_a_graph_that_has_an_async_node() -> None:
    """
    Arrange: a graph with an async node, with and without a checkpointer.
    Act: stream a run with the sync function.
    Assert: each call raises TypeError.
    """
    builder = build_builder(("async", write_async))

    with pytest.raises(TypeError):
        builder.compile().stream({})
    with pytest.raises(TypeError):
        builder.compile(checkpointer=InMemoryCheckpointer()).stream({}, {"thread_id": "thread"})


@pytest.mark.parametrize("sync", [False, True])
def test_stream_raises_the_exception_of_a_node_after_the_parts_before_it(*, sync: bool) -> None:
    """
    Arrange: a graph in which the second node raises.
    Act: read a stream to its end, with the async function or the sync function.
    Assert: the stream gives the update of the first node, then raises the exception of the second node.
    """
    graph = build_builder(("first", build_writer("first")), ("fail", fail)).compile()
    actual: list[UpdatesPart] = []

    async def read() -> None:
        stream = graph.astream({})
        actual.append(await anext(part async for part in stream if isinstance(part, UpdatesPart)))
        await collect(stream)

    def read_sync() -> None:
        stream = graph.stream({})
        actual.append(next(part for part in stream if isinstance(part, UpdatesPart)))
        list(stream)

    with pytest.raises(ConnectionError):
        if sync:
            read_sync()
        else:
            asyncio.run(read())

    expected = [UpdatesPart("first", {"notes": ["first"]})]
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_stream_that_the_reader_leaves_cancels_the_run() -> None:
    """
    Arrange: a graph of two nodes in a chain.
    Act: read the first part of a stream, then close the stream.
    Assert: the second node does not run.
    """
    calls: list[str] = []

    def record(state: State) -> State:
        calls.append("second")
        return {}

    graph = build_builder(("first", build_writer("first")), ("second", record)).compile()

    async def read_one_part() -> None:
        stream = graph.astream({})
        async for _ in stream:
            break
        await stream.aclose()

    asyncio.run(read_one_part())

    assert calls == [], f"Expected no call of the second node, got {calls}."


def test_stream_of_a_thread_gives_the_interrupt_and_continues_with_a_resume_value() -> None:
    """
    Arrange: a graph with a checkpointer whose second node calls interrupt.
    Act: stream a run, then stream a run with a resume value.
    Assert: the first stream ends with a values part and an interrupts part that have the interrupt, and the
    second stream has the update with the answer.
    """
    graph = build_builder(("first", build_writer("first")), ("ask", ask)).compile(checkpointer=InMemoryCheckpointer())

    stopped = asyncio.run(collect(graph.astream({}, {"thread_id": "thread"})))
    resumed = list(graph.stream(Command(resume="answer"), {"thread_id": "thread"}))

    values = [part.interrupts[0].value for part in stopped[-2:] if isinstance(part, ValuesPart | InterruptsPart)]
    kinds = [type(part) for part in stopped[-2:]]
    updates = [part for part in resumed if isinstance(part, UpdatesPart)]
    assert kinds == [ValuesPart, InterruptsPart], f"Expected a values part and an interrupts part, got {kinds}."
    assert values == ["question", "question"], f"Expected the value of the interrupt in each part, got {values}."
    assert updates == [UpdatesPart("ask", {"notes": ["answer"]})], f"Expected the answer, got {updates}."


def test_stream_without_a_checkpointer_rejects_an_interrupt() -> None:
    """
    Arrange: a graph without a checkpointer whose node calls interrupt.
    Act: stream a run.
    Assert: the stream raises RuntimeError.
    """
    graph = build_builder(("ask", ask)).compile()

    with pytest.raises(RuntimeError):
        list(graph.stream({}))


def test_stream_writer_outside_a_run_is_rejected() -> None:
    """
    Arrange: no running graph.
    Act: ask for the stream writer.
    Assert: the call raises RuntimeError.
    """
    with pytest.raises(RuntimeError):
        get_stream_writer()


def test_invoke_ignores_the_data_that_a_node_writes() -> None:
    """
    Arrange: a graph whose nodes write to the stream writer.
    Act: run the graph with invoke.
    Assert: the output has the notes of the nodes.
    """
    actual = build_chain().invoke({})

    assert actual == Output(notes=["first", "second"]), f"Expected the notes of two nodes, got {actual}."


def test_task_parts_give_the_start_and_the_result_of_each_node() -> None:
    """
    Arrange: a graph of two nodes in a chain.
    Act: stream a run.
    Assert: each node has a start part with its input and then a result part with its update and the same id.
    """
    graph = build_chain()

    parts = list(graph.stream({"notes": ["input"]}))

    actual = [part for part in parts if isinstance(part, TaskStartPart | TaskResultPart)]
    starts = [part for part in actual if isinstance(part, TaskStartPart)]
    results = [part for part in actual if isinstance(part, TaskResultPart)]
    kinds = [(type(part), part.name) for part in actual]
    expected_kinds = [
        (TaskStartPart, "first"),
        (TaskResultPart, "first"),
        (TaskStartPart, "second"),
        (TaskResultPart, "second"),
    ]
    assert kinds == expected_kinds, f"Expected {expected_kinds}, got {kinds}."
    assert [part.input for part in starts] == [{"notes": ["input"]}, {"notes": ["input", "first"]}], (
        f"Expected the input of each node, got {starts}."
    )
    assert [part.update for part in results] == [{"notes": ["first"]}, {"notes": ["second"]}], (
        f"Expected the update of each node, got {results}."
    )
    assert [part.id for part in starts] == [part.id for part in results], "Expected the same id for start and result."
    assert len({part.id for part in starts}) == 2, f"Expected a different id for each task, got {starts}."


def test_task_result_part_gives_the_interrupts_of_a_node_that_stops() -> None:
    """
    Arrange: a graph with a checkpointer whose node calls interrupt.
    Act: stream a run.
    Assert: the result part of the node has the interrupt and no update.
    """
    graph = build_builder(("ask", ask)).compile(checkpointer=InMemoryCheckpointer())

    actual = list(graph.stream({}, {"thread_id": "thread"}))

    results = [part for part in actual if isinstance(part, TaskResultPart)]
    values = [[item.value for item in part.interrupts] for part in results]
    assert values == [["question"]], f"Expected one result part with the interrupt, got {actual}."
    assert [part.update for part in results] == [{}], f"Expected no update, got {results}."


def test_checkpoint_parts_give_the_state_of_each_checkpoint() -> None:
    """
    Arrange: a graph of two nodes in a chain with a checkpointer.
    Act: stream a run.
    Assert: the checkpoint parts have the state of each checkpoint in the order of the steps, and the last is
    the thread state.
    """
    graph = build_builder(("first", build_writer("first")), ("second", build_writer("second"))).compile(
        checkpointer=InMemoryCheckpointer()
    )

    parts = asyncio.run(collect(graph.astream({}, {"thread_id": "thread"})))

    actual = [part for part in parts if isinstance(part, CheckpointPart)]
    steps = [(part.state.step, part.state.next) for part in actual]
    expected = [(-1, ("__start__",)), (0, ("first",)), (1, ("second",)), (2, ())]
    assert steps == expected, f"Expected {expected}, got {steps}."
    assert actual[-1].state == graph.get_state({"thread_id": "thread"}), (
        "Expected the state of the thread in the last part."
    )


def test_stream_gives_no_checkpoint_part_for_a_run_that_continues_without_a_new_checkpoint() -> None:
    """
    Arrange: a thread that stopped at an interrupt after its first node.
    Act: resume the thread with a stream.
    Assert: the stream has only the checkpoint that the resumed node made.
    """
    graph = build_builder(("first", build_writer("first")), ("ask", ask)).compile(checkpointer=InMemoryCheckpointer())
    graph.invoke({}, {"thread_id": "thread"})

    actual = list(graph.stream(Command(resume="answer"), {"thread_id": "thread"}))

    steps = [part.state.step for part in actual if isinstance(part, CheckpointPart)]
    assert steps == [2], f"Expected only the checkpoint of step 2, got {steps}."


def test_stream_without_a_checkpointer_gives_no_checkpoint_part() -> None:
    """
    Arrange: a graph without a checkpointer.
    Act: stream a run.
    Assert: the stream has no checkpoint part.
    """
    graph = build_chain()

    actual = list(graph.stream({}))

    checkpoints = [part for part in actual if isinstance(part, CheckpointPart)]
    assert checkpoints == [], f"Expected no checkpoint part, got {checkpoints}."

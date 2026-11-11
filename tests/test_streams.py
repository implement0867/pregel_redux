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
    StreamMode,
    get_stream_writer,
    interrupt,
)
from pregel_redux.streams import CustomPart, InterruptsPart, StreamPart, UpdatesPart, ValuesPart
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


def test_values_mode_gives_the_state_after_each_step() -> None:
    """
    Arrange: a graph of two nodes in a chain.
    Act: stream a run in the values mode.
    Assert: the stream has the state after the input and after each node.
    """
    graph = build_chain()

    actual = asyncio.run(collect(graph.astream({"notes": ["input"]}, stream_mode=StreamMode.values)))

    expected = [
        ValuesPart((), {"notes": ["input"]}, ()),
        ValuesPart((), {"notes": ["input", "first"]}, ()),
        ValuesPart((), {"notes": ["input", "first", "second"]}, ()),
    ]
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_updates_mode_gives_the_update_of_each_node() -> None:
    """
    Arrange: a graph of two nodes in a chain.
    Act: stream a run in the updates mode.
    Assert: the stream has one part for each node with the update of that node.
    """
    graph = build_chain()

    actual = asyncio.run(collect(graph.astream({"notes": ["input"]}, stream_mode=StreamMode.updates)))

    expected = [UpdatesPart((), "first", {"notes": ["first"]}), UpdatesPart((), "second", {"notes": ["second"]})]
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_custom_mode_gives_the_data_that_a_node_writes() -> None:
    """
    Arrange: a graph with a sync node and an async node that write to the stream writer.
    Act: stream a run in the custom mode.
    Assert: the stream has the data of each node in the order of the nodes.
    """
    graph = build_builder(("first", build_writer("first")), ("async", write_async)).compile()

    actual = asyncio.run(collect(graph.astream({}, stream_mode=StreamMode.custom)))

    expected = [CustomPart((), "first works"), CustomPart((), "async works")]
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_several_modes_give_the_parts_of_each_mode_in_the_order_of_the_run() -> None:
    """
    Arrange: a graph with one node.
    Act: stream a run in the custom, updates and values modes.
    Assert: the stream has the parts of all modes in the order in which the run made them.
    """
    graph = build_builder(("first", build_writer("first"))).compile()
    modes = [StreamMode.custom, StreamMode.updates, StreamMode.values]

    actual = asyncio.run(collect(graph.astream({}, stream_mode=modes)))

    expected: list[StreamPart] = [
        ValuesPart((), {"notes": []}, ()),
        CustomPart((), "first works"),
        UpdatesPart((), "first", {"notes": ["first"]}),
        ValuesPart((), {"notes": ["first"]}, ()),
    ]
    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize(
    ("subgraphs", "expected"),
    [
        (False, [((), "child"), ((), "last")]),
        (True, [(("child",), "inner"), ((), "child"), ((), "last")]),
    ],
)
def test_parts_of_a_child_graph_are_in_the_stream_only_when_subgraphs_is_set(
    *, subgraphs: bool, expected: list[tuple[tuple[str, ...], str]]
) -> None:
    """
    Arrange: a parent graph with a child graph that has one node.
    Act: stream a run in the updates mode, with and without subgraphs.
    Assert: the parts of the child graph have the namespace of the child and are present only with subgraphs.
    """
    graph = build_parent()

    parts = asyncio.run(collect(graph.astream({}, stream_mode=StreamMode.updates, subgraphs=subgraphs)))

    actual = [(part.ns, part.node) for part in parts if isinstance(part, UpdatesPart)]
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_sync_stream_gives_the_parts_of_the_async_stream() -> None:
    """
    Arrange: a graph of two sync nodes in a chain.
    Act: stream a run with the sync function, outside and inside a running event loop.
    Assert: each stream is equal to the async stream.
    """
    graph = build_chain()
    modes = [StreamMode.custom, StreamMode.updates, StreamMode.values]
    expected = asyncio.run(collect(graph.astream({}, stream_mode=modes)))

    async def stream_inside_a_loop() -> list[StreamPart]:
        return list(graph.stream({}, stream_mode=modes))

    actual = list(graph.stream({}, stream_mode=modes))
    inside = asyncio.run(stream_inside_a_loop())

    assert actual == expected, f"Expected {expected}, got {actual}."
    assert inside == expected, f"Expected {expected}, got {inside}."


def test_sync_stream_rejects_a_graph_that_has_an_async_node() -> None:
    """
    Arrange: a graph with an async node, with and without a checkpointer.
    Act: stream a run with the sync function.
    Assert: each call raises TypeError that names astream.
    """
    builder = build_builder(("async", write_async))

    with pytest.raises(TypeError, match="astream"):
        builder.compile().stream({}, stream_mode=StreamMode.values)
    with pytest.raises(TypeError, match="astream"):
        builder.compile(checkpointer=InMemoryCheckpointer()).stream(
            {}, thread_id="thread", stream_mode=StreamMode.values
        )


@pytest.mark.parametrize("sync", [False, True])
def test_stream_raises_the_exception_of_a_node_after_the_parts_before_it(*, sync: bool) -> None:
    """
    Arrange: a graph in which the second node raises.
    Act: stream a run in the updates mode, with the async function or the sync function.
    Assert: the stream gives the part of the first node, then raises the exception of the second node.
    """
    graph = build_builder(("first", build_writer("first")), ("fail", fail)).compile()
    actual: list[object] = []

    async def read() -> None:
        stream = graph.astream({}, stream_mode=StreamMode.updates)
        actual.append(await anext(stream))
        await anext(stream)

    def read_sync() -> None:
        stream = graph.stream({}, stream_mode=StreamMode.updates)
        actual.append(next(stream))
        next(stream)

    with pytest.raises(ConnectionError, match="no network"):
        read_sync() if sync else asyncio.run(read())

    expected = [UpdatesPart((), "first", {"notes": ["first"]})]
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
        stream = graph.astream({}, stream_mode=StreamMode.updates)
        async for _ in stream:
            break
        await stream.aclose()

    asyncio.run(read_one_part())

    assert calls == [], f"Expected no call of the second node, got {calls}."


def test_stream_of_a_thread_gives_the_interrupt_and_continues_with_a_resume_value() -> None:
    """
    Arrange: a graph with a checkpointer whose second node calls interrupt.
    Act: stream a run in the updates and values modes, then stream a run with a resume value.
    Assert: the first stream ends with the interrupt in each mode, and the second stream has the answer.
    """
    graph = build_builder(("first", build_writer("first")), ("ask", ask)).compile(checkpointer=InMemoryCheckpointer())
    modes = [StreamMode.updates, StreamMode.values]

    stopped = asyncio.run(collect(graph.astream({}, thread_id="thread", stream_mode=modes)))
    resumed = list(graph.stream(Command(resume="answer"), thread_id="thread", stream_mode=StreamMode.updates))

    values = [part.interrupts[0].value for part in stopped[-2:] if isinstance(part, ValuesPart | InterruptsPart)]
    kinds = [type(part) for part in stopped[-2:]]
    assert kinds == [ValuesPart, InterruptsPart], f"Expected a values part and an interrupts part, got {kinds}."
    assert values == ["question", "question"], f"Expected the value of the interrupt in each part, got {values}."
    assert resumed == [UpdatesPart((), "ask", {"notes": ["answer"]})], f"Expected the answer, got {resumed}."


def test_stream_without_a_checkpointer_rejects_an_interrupt() -> None:
    """
    Arrange: a graph without a checkpointer whose node calls interrupt.
    Act: stream a run.
    Assert: the stream raises RuntimeError that names the checkpointer.
    """
    graph = build_builder(("ask", ask)).compile()

    with pytest.raises(RuntimeError, match="checkpointer"):
        list(graph.stream({}, stream_mode=StreamMode.values))


def test_stream_writer_outside_a_run_is_rejected() -> None:
    """
    Arrange: no running graph.
    Act: ask for the stream writer.
    Assert: the call raises RuntimeError.
    """
    with pytest.raises(RuntimeError, match="inside a node"):
        get_stream_writer()


def test_invoke_ignores_the_data_that_a_node_writes() -> None:
    """
    Arrange: a graph whose nodes write to the stream writer.
    Act: run the graph with invoke.
    Assert: the output has the notes of the nodes.
    """
    actual = build_chain().invoke({})

    assert actual == Output(notes=["first", "second"]), f"Expected the notes of two nodes, got {actual}."

import asyncio
import threading
from collections.abc import Callable, Iterator
from functools import partial
from typing import Annotated, TypedDict

import pytest

from pregel_redux import Depends, Endpoint, Reducer, Scope, StateGraph
from tests.conftest import Journal, join_lists


class State(TypedDict, total=False):
    released: Annotated[list[bool], Reducer(join_lists, initial_factory=list)]


def build_blocked_node(wait_for_release: Callable[[], bool]) -> Callable[..., object]:
    def wait_in_node(state: State) -> State:
        return {"released": [wait_for_release()]}

    return wait_in_node


def build_node_with_blocked_provider(wait_for_release: Callable[[], bool], *, scope: Scope) -> Callable[..., object]:
    async def read_provided(state: State, released: Annotated[bool, Depends(wait_for_release, scope=scope)]) -> State:
        return {"released": [released]}

    return read_provided


def build_node_with_blocked_generator_provider(
    wait_for_release: Callable[[], bool], *, scope: Scope
) -> Callable[..., object]:
    def provide_released() -> Iterator[bool]:
        yield wait_for_release()

    async def read_provided(state: State, released: Annotated[bool, Depends(provide_released, scope=scope)]) -> State:
        return {"released": [released]}

    return read_provided


@pytest.mark.parametrize(
    "build_blocked",
    [
        build_blocked_node,
        partial(build_node_with_blocked_provider, scope=Scope.node),
        partial(build_node_with_blocked_provider, scope=Scope.run),
        partial(build_node_with_blocked_generator_provider, scope=Scope.node),
        partial(build_node_with_blocked_generator_provider, scope=Scope.run),
    ],
    ids=["node", "node-scoped-provider", "run-scoped-provider", "node-scoped-generator", "run-scoped-generator"],
)
def test_sync_function_that_blocks_lets_an_async_node_of_the_same_step_run(
    build_blocked: Callable[[Callable[[], bool]], Callable[..., object]],
) -> None:
    """
    Arrange: a step with a sync node or sync provider that waits for an event, and an async node that sets the event.
    Act: run the graph with ainvoke.
    Assert: the sync function received the event before its timeout.
    """
    release = threading.Event()

    async def release_blocked(state: State) -> None:
        release.set()

    builder = StateGraph(State)
    builder.add_node("blocked", build_blocked(lambda: release.wait(timeout=1)))
    builder.add_node("release", release_blocked)
    builder.add_edge(Endpoint.start, "blocked")
    builder.add_edge(Endpoint.start, "release")

    actual = asyncio.run(builder.compile().ainvoke({}))

    assert actual == {"released": [True]}, (
        f"Expected the async node to release the sync function, got {actual}. "
        "A sync function must not block the event loop."
    )


@pytest.mark.parametrize("count", [2, 3, 5])
def test_sync_nodes_of_one_step_run_at_the_same_time(count: int) -> None:
    """
    Arrange: a graph with several sync entry nodes that wait for each other at a barrier.
    Act: run the graph with ainvoke.
    Assert: each node passed the barrier.
    """
    barrier = threading.Barrier(count, timeout=1)

    def meet_other_nodes(state: State) -> State:
        barrier.wait()
        return {"released": [True]}

    builder = StateGraph(State)
    for index in range(count):
        builder.add_node(f"node-{index}", meet_other_nodes)
        builder.add_edge(Endpoint.start, f"node-{index}")
    expected: State = {"released": [True] * count}

    actual = asyncio.run(builder.compile().ainvoke({}))

    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize("released", [[], [True], [True, False]], ids=repr)
def test_invoke_gives_the_output_of_ainvoke(released: list[bool]) -> None:
    """
    Arrange: a compiled graph with one sync node, and the output of ainvoke for an input.
    Act: call invoke with the same input.
    Assert: the output equals the output of ainvoke.
    """
    builder = StateGraph(State)
    builder.add_node("write", lambda state: {"released": [True]})
    builder.add_edge(Endpoint.start, "write")
    graph = builder.compile()
    expected = asyncio.run(graph.ainvoke({"released": released}))

    actual = graph.invoke({"released": released})

    assert actual == expected, f"Expected {expected}, got {actual}. invoke must give the output of ainvoke."


@pytest.mark.parametrize("depth", [0, 1, 2], ids=["node-of-the-graph", "node-of-a-child", "node-of-a-grandchild"])
@pytest.mark.parametrize("sync_nodes", [0, 2])
def test_invoke_rejects_a_graph_that_has_an_async_node(journal: Journal, depth: int, sync_nodes: int) -> None:
    """
    Arrange: a graph with an async node at a given depth of child graphs, with or without sync nodes.
    Act: call invoke.
    Assert: a TypeError is raised and no node ran.
    """

    async def write_async(state: State) -> None:
        journal.events.append("async node")

    graph = StateGraph(State)
    graph.add_node("write", write_async)
    graph.add_edge(Endpoint.start, "write")
    for _ in range(depth):
        parent = StateGraph(State)
        parent.add_node("child", graph)
        parent.add_edge(Endpoint.start, "child")
        graph = parent
    for index in range(sync_nodes):
        graph.add_node(f"sync-{index}", lambda state: journal.events.append("sync node"))
        graph.add_edge(Endpoint.start, f"sync-{index}")

    with pytest.raises(TypeError):
        graph.compile().invoke({})

    assert journal.events == [], f"Expected no node to run, got {journal.events}."


@pytest.mark.parametrize("step_limit", [1, 2])
def test_invoke_applies_the_step_limit(step_limit: int) -> None:
    """
    Arrange: a chain of nodes that is one step longer than the step limit.
    Act: call invoke with the step limit.
    Assert: a RuntimeError is raised.
    """
    builder = StateGraph(State)
    for index in range(step_limit + 1):
        builder.add_node(f"node-{index}", lambda state: None)
        builder.add_edge(Endpoint.start if index == 0 else f"node-{index - 1}", f"node-{index}")

    with pytest.raises(RuntimeError):
        builder.compile().invoke({}, step_limit=step_limit)


@pytest.mark.parametrize("released", [[], [True], [True, False]], ids=repr)
def test_invoke_inside_a_running_event_loop_gives_the_output(released: list[bool]) -> None:
    """
    Arrange: a compiled graph with one sync node, and a coroutine that calls invoke.
    Act: run the coroutine on an event loop.
    Assert: invoke returned the output of the graph.
    """
    builder = StateGraph(State)
    builder.add_node("write", lambda state: {"released": [True]})
    builder.add_edge(Endpoint.start, "write")
    graph = builder.compile()
    expected: State = {"released": [*released, True]}

    async def call_invoke() -> State:
        return graph.invoke({"released": released})

    actual = asyncio.run(call_invoke())

    assert actual == expected, f"Expected {expected}, got {actual}. invoke must work inside a running event loop."


@pytest.mark.parametrize("scope", list(Scope), ids=lambda scope: scope.name)
@pytest.mark.parametrize("error_type", [LookupError, ValueError])
def test_node_with_a_sync_provider_finishes_when_another_node_fails(
    journal: Journal, error_type: type[Exception], scope: Scope
) -> None:
    """
    Arrange: a sync generator provider that is in its open step while another node of the step fails.
    Act: run the graph with ainvoke.
    Assert: the failure is raised after the node of the provider ran and the provider closed.
    """
    opened = threading.Event()
    release = threading.Event()

    def open_connection() -> Iterator[int]:
        number = journal.open_resource("link")
        opened.set()
        release.wait(timeout=1)
        try:
            yield number
        finally:
            journal.events.append(f"close {number}")

    def use_value(state: State, number: Annotated[int, Depends(open_connection, scope=scope)]) -> None:
        journal.events.append(f"node {number}")

    async def raise_error(state: State) -> None:
        await asyncio.to_thread(opened.wait, 1)
        asyncio.get_running_loop().call_later(0.05, release.set)
        raise error_type("no value")

    builder = StateGraph(State)
    builder.add_node("use", use_value)
    builder.add_node("fail", raise_error)
    builder.add_edge(Endpoint.start, "use")
    builder.add_edge(Endpoint.start, "fail")
    expected = ["open link 1", "node 1", "close 1"]

    with pytest.raises(error_type):
        asyncio.run(builder.compile().ainvoke({}))

    assert journal.events == expected, (
        f"Expected the events {expected}, got {journal.events}. "
        "The other nodes of a step run to their end after a failure, and their providers close."
    )


@pytest.mark.parametrize("nodes_after_the_break", [1, 2])
def test_stream_stops_the_run_when_the_reader_stops(journal: Journal, nodes_after_the_break: int) -> None:
    """
    Arrange: a chain of sync nodes, and a reader of stream that received the update of the first node.
    Act: close the stream while the second node runs.
    Assert: no node after the second node ran.
    """
    started = threading.Event()
    release = threading.Event()

    def wait_in_node(state: State) -> None:
        started.set()
        release.wait(timeout=1)
        journal.events.append("second")

    builder = StateGraph(State)
    builder.add_node("first", lambda state: journal.events.append("first"))
    builder.add_node("second", wait_in_node)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "second")
    for index in range(nodes_after_the_break):
        builder.add_node(f"later-{index}", lambda state: journal.events.append("later"))
        builder.add_edge("second" if index == 0 else f"later-{index - 1}", f"later-{index}")
    parts = builder.compile().stream({})
    next(parts)
    started.wait(timeout=1)
    timer = threading.Timer(0.05, release.set)
    expected = ["first", "second"]

    timer.start()
    parts.close()

    assert journal.events == expected, (
        f"Expected the events {expected}, got {journal.events}. "
        "A reader that stops must stop the run after the nodes that run at that time."
    )

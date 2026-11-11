import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from itertools import pairwise
from typing import Annotated, TypedDict

import pytest

from pregel_redux import Command, Depends, Endpoint, Reducer, Scope, Send, StateGraph
from tests.conftest import Journal, append_item, join_lists


class State(TypedDict, total=False):
    topics: list[str]
    jokes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]
    visits: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class TopicsInput(TypedDict, total=False):
    topics: list[str]


class JokesOutput(TypedDict, total=False):
    jokes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class AppendState(TypedDict, total=False):
    items: Annotated[list[str], Reducer(append_item, initial_factory=list)]


def build_joke_writer(joke: str) -> Callable[[State], JokesOutput]:
    return lambda state: {"jokes": [joke]}


def build_chain_graph(names: list[str]) -> StateGraph[State, TopicsInput, JokesOutput]:
    builder = StateGraph(State, input_schema=TopicsInput, output_schema=JokesOutput)
    for name in names:
        builder.add_node(name, build_joke_writer(name))
    builder.add_edge(Endpoint.start, names[0])
    for source, target in pairwise(names):
        builder.add_edge(source, target)
    return builder


def build_reader_graph(read_value: Callable[..., object], name: str) -> StateGraph[State, TopicsInput, JokesOutput]:
    builder = StateGraph(State, input_schema=TopicsInput, output_schema=JokesOutput)
    builder.add_node(name, read_value)
    builder.add_edge(Endpoint.start, name)
    return builder


def nest_graph(
    child: StateGraph[State, TopicsInput, JokesOutput], depth: int
) -> StateGraph[State, TopicsInput, JokesOutput]:
    graph = child
    for _ in range(depth):
        parent = StateGraph(State, input_schema=TopicsInput, output_schema=JokesOutput)
        parent.add_node("child", graph)
        parent.add_edge(Endpoint.start, "child")
        graph = parent
    return graph


@pytest.mark.parametrize(
    ("names", "depth"),
    [(["first"], 1), (["first", "second"], 1), (["first", "second", "third"], 1), (["first", "second"], 3)],
    ids=["one-node", "two-nodes", "three-nodes", "three-levels"],
)
def test_child_graph_runs_as_a_node_of_its_parent(names: list[str], depth: int) -> None:
    """
    Arrange: a parent graph with a node before and a node after a child graph of a given depth.
    Act: run the parent graph.
    Assert: the jokes of the child nodes are between the jokes of the parent nodes.
    """
    builder = StateGraph(State)
    builder.add_node("before", lambda state: {"jokes": ["before"]})
    builder.add_node("child", nest_graph(build_chain_graph(names), depth - 1))
    builder.add_node("after", lambda state: {"jokes": ["after"]})
    builder.add_edge(Endpoint.start, "before")
    builder.add_edge("before", "child")
    builder.add_edge("child", "after")
    expected = ["before", *names, "after"]

    actual = asyncio.run(builder.compile().ainvoke({}))

    assert actual.get("jokes", []) == expected, f"Expected the jokes {expected}, got {actual}."


@pytest.mark.parametrize(
    ("input_state", "expected"),
    [({}, ["node"]), ({"items": "input"}, ["input", "node"])],
    ids=["without-input", "with-input"],
)
def test_child_graph_starts_with_the_channel_values_of_its_parent(input_state: object, expected: list[str]) -> None:
    """
    Arrange: a parent and a child graph with a reducer whose update type is not its channel type.
    Act: run the parent graph, which writes to the channel and then calls the child.
    Assert: the child read the value of the parent, and the reducer of the child did not combine it again.
    """
    received: list[AppendState] = []
    child = StateGraph(AppendState, output_schema=TopicsInput)
    child.add_node("receive", received.append)
    child.add_edge(Endpoint.start, "receive")
    builder = StateGraph[object, object, object](AppendState)
    builder.add_node("write", lambda state: {"items": "node"})
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "write")
    builder.add_edge("write", "child")

    asyncio.run(builder.compile().ainvoke(input_state))

    assert received == [{"items": expected}], f"Expected the child to read {expected}, got {received}."


@pytest.mark.parametrize(
    ("input_state", "expected"),
    [
        ({"topics": ["cats"], "visits": ["input"]}, {"topics": ["cats"]}),
        ({"visits": ["input"]}, {}),
        ({}, {}),
    ],
    ids=["field-with-a-value", "field-without-a-value", "empty-state"],
)
def test_child_graph_receives_the_fields_of_its_input_type(input_state: State, expected: State) -> None:
    """
    Arrange: a child graph with an input type that has one field of the parent state.
    Act: run the parent graph with an input state.
    Assert: the child received only the fields of its input type and the initial values of its own reducers.
    """
    received: list[State] = []
    child = StateGraph(State, input_schema=TopicsInput)
    child.add_node("receive", received.append)
    child.add_edge(Endpoint.start, "receive")
    builder = StateGraph(State)
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "child")

    asyncio.run(builder.compile().ainvoke(input_state))

    assert received == [{"jokes": [], "visits": [], **expected}], f"Expected {expected}, got {received}."


@pytest.mark.parametrize("visits", [[], ["input"], ["input", "other"]], ids=repr)
def test_child_graph_writes_the_fields_of_its_output_type(visits: list[str]) -> None:
    """
    Arrange: a child graph whose node writes two channels and whose output type has one of them.
    Act: run the parent graph.
    Assert: the parent state has only the write for the output field.
    """
    child = StateGraph(State, output_schema=JokesOutput)
    child.add_node("write", lambda state: {"jokes": ["joke"], "visits": ["child"]})
    child.add_edge(Endpoint.start, "write")
    builder = StateGraph(State)
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "child")
    expected: State = {"visits": visits, "jokes": ["joke"]}

    actual = asyncio.run(builder.compile().ainvoke({"visits": visits}))

    assert actual == expected, f"Expected {expected}, got {actual}. The child must write only its output fields."


@pytest.mark.parametrize(
    ("jokes", "expected"),
    [([], ["joke"]), (["input"], ["input", "input", "joke"])],
    ids=["channel-without-a-value", "channel-with-a-value"],
)
def test_parent_reduces_the_output_of_a_child_graph_that_also_received_the_channel(
    jokes: list[str], expected: list[str]
) -> None:
    """
    Arrange: a child graph that receives and returns a channel that has a reducer.
    Act: run the parent graph with a value for the channel.
    Assert: the parent reduced the complete output of the child into the channel.
    """
    child = StateGraph(State)
    child.add_node("write", lambda state: {"jokes": ["joke"]})
    child.add_edge(Endpoint.start, "write")
    builder = StateGraph(State)
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "child")

    actual = asyncio.run(builder.compile().ainvoke({"jokes": jokes}))

    assert actual.get("jokes", []) == expected, (
        f"Expected the jokes {expected}, got {actual}. The output of a child is a write that the parent reduces."
    )


@pytest.mark.parametrize(
    ("topics", "expected"),
    [([], []), (["cats"], ["cats"]), (["cats", "dogs"], ["cats", "dogs"])],
    ids=["no-send", "one-send", "two-sends"],
)
def test_each_send_starts_the_child_graph_with_its_payload(topics: list[str], expected: list[str]) -> None:
    """
    Arrange: a parent node that sends one payload for each topic to a child graph.
    Act: run the parent graph with the topics.
    Assert: the child wrote one joke for each payload.
    """
    child = StateGraph(State, input_schema=TopicsInput, output_schema=JokesOutput)
    child.add_node("write", lambda state: {"jokes": state.get("topics", [])})
    child.add_edge(Endpoint.start, "write")
    builder = StateGraph(State)
    builder.add_node(
        "plan",
        lambda state: Command[str](goto=[Send("child", {"topics": [topic]}) for topic in state.get("topics", [])]),
    )
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "plan")

    actual = asyncio.run(builder.compile().ainvoke({"topics": topics}))

    assert actual.get("jokes", []) == expected, f"Expected the jokes {expected}, got {actual}."


@pytest.mark.parametrize(
    "payload",
    [{"missing": 1}, {"topics": ["cats"], "missing": 1}],
    ids=["unknown-field", "known-and-unknown-field"],
)
def test_send_with_a_field_that_is_not_a_channel_of_the_child_graph_is_rejected(payload: dict[str, object]) -> None:
    """
    Arrange: a parent node that sends a payload with a field that the child graph does not have.
    Act: run the parent graph.
    Assert: a ValueError is raised.
    """
    child = StateGraph(State, input_schema=TopicsInput, output_schema=JokesOutput)
    child.add_node("write", lambda state: None)
    child.add_edge(Endpoint.start, "write")
    builder = StateGraph(State)
    builder.add_node("plan", lambda state: Command[str](goto=[Send("child", payload)]))
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "plan")

    with pytest.raises(ValueError):
        asyncio.run(builder.compile().ainvoke({}))


@pytest.mark.parametrize(
    ("first_depth", "second_depth", "same_step"),
    [(0, 1, False), (1, 1, False), (1, 1, True), (0, 3, False), (2, 3, True)],
    ids=[
        "parent-then-child",
        "child-then-child",
        "two-children-in-one-step",
        "parent-then-deep-child",
        "deep-children",
    ],
)
def test_child_graph_shares_the_run_scoped_values_of_its_parent(
    journal: Journal, first_depth: int, second_depth: int, *, same_step: bool
) -> None:
    """
    Arrange: two nodes at given depths of child graphs that request one run-scoped provider, in one step or in two.
    Act: run the parent graph.
    Assert: the provider opened one time and closed one time.
    """

    async def provide_resource() -> AsyncGenerator[int]:
        await asyncio.sleep(0)
        yield journal.open_resource("resource")
        journal.events.append("close resource")

    def read_resource(
        state: State, resource: Annotated[int, Depends(provide_resource, scope=Scope.run)]
    ) -> JokesOutput:
        journal.events.append("read resource")
        return {"jokes": [str(resource)]}

    builder = StateGraph(State)
    builder.add_node("first", nest_graph(build_reader_graph(read_resource, "read"), first_depth))
    builder.add_node("second", nest_graph(build_reader_graph(read_resource, "read"), second_depth))
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start if same_step else "first", "second")
    expected = ["open resource 1", "read resource", "read resource", "close resource"]

    asyncio.run(builder.compile().ainvoke({}))

    assert journal.events == expected, (
        f"Expected the events {expected}, got {journal.events}. A run-scoped provider must open one time for a run."
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("replacement", ["fake", "other"])
def test_overrides_of_the_parent_apply_to_a_child_graph(depth: int, replacement: str) -> None:
    """
    Arrange: a node at a given depth of child graphs that requests a provider.
    Act: compile the parent with an override of the provider and run it.
    Assert: the node received the value of the override.
    """

    def provide_name() -> str:
        return "real"

    def write_name(state: State, name: Annotated[str, Depends(provide_name)]) -> JokesOutput:
        return {"jokes": [name]}

    builder = nest_graph(build_reader_graph(write_name, "write"), depth)

    actual = asyncio.run(builder.compile(dependency_overrides={provide_name: lambda: replacement}).ainvoke({}))

    assert actual.get("jokes", []) == [replacement], f"Expected the child to receive {replacement!r}, got {actual}."


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_dependencies_of_a_graph_include_those_of_its_child_graphs(depth: int) -> None:
    """
    Arrange: a node at a given depth of child graphs that requests a provider.
    Act: read the dependencies of the parent graph.
    Assert: the dependencies contain the provider.
    """

    def provide_name() -> str:
        return "real"

    def write_name(state: State, name: Annotated[str, Depends(provide_name)]) -> JokesOutput:
        return {"jokes": [name]}

    builder = nest_graph(build_reader_graph(write_name, "write"), depth)

    actual = builder.dependencies

    assert actual == {provide_name}, f"Expected the dependencies to be {{provide_name}}, got {actual}."


@pytest.mark.parametrize(
    ("child_steps", "step_limit", "expectation"),
    [
        (1, 1, does_not_raise()),
        (3, 3, does_not_raise()),
        (4, 3, pytest.raises(RuntimeError)),
        (2, 1, pytest.raises(RuntimeError)),
    ],
    ids=["one-step-at-the-limit", "three-steps-at-the-limit", "four-steps-over-the-limit", "two-steps-over-the-limit"],
)
@pytest.mark.parametrize("depth", [1, 2])
def test_child_graph_counts_its_own_steps_against_the_step_limit_of_the_run(
    child_steps: int, step_limit: int, depth: int, expectation: AbstractContextManager[object]
) -> None:
    """
    Arrange: a child graph with a given number of steps at a given depth.
    Act: run the parent graph with a step limit.
    Assert: the run finishes at the limit and raises a RuntimeError above it.
    """
    names = [f"node-{index}" for index in range(child_steps)]
    graph = nest_graph(build_chain_graph(names), depth).compile()

    with expectation:
        asyncio.run(graph.ainvoke({}, step_limit=step_limit))


@pytest.mark.parametrize("name", ["child", "other"])
def test_child_graph_with_the_name_of_a_node_is_rejected(name: str) -> None:
    """
    Arrange: a graph that has a node with a name.
    Act: add a child graph with the same name.
    Assert: a ValueError is raised.
    """
    builder = StateGraph(State)
    builder.add_node(name, lambda state: None)

    with pytest.raises(ValueError):
        builder.add_node(name, build_chain_graph(["first"]))

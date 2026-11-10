import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from itertools import pairwise
from operator import add
from typing import Annotated, TypedDict

import pytest

from pregel_redux import Command, Depends, Endpoint, Scope, Send, StateGraph
from tests.conftest import Journal


class State(TypedDict, total=False):
    topics: list[str]
    jokes: Annotated[list[str], add]
    visits: Annotated[list[str], add]


class TopicsInput(TypedDict, total=False):
    topics: list[str]


class JokesOutput(TypedDict, total=False):
    jokes: Annotated[list[str], add]


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
    [
        ({"topics": ["cats"], "visits": ["input"]}, {"topics": ["cats"]}),
        ({"visits": ["input"]}, {}),
        ({}, {}),
    ],
    ids=["field-with-a-value", "field-without-a-value", "empty-state"],
)
def test_child_graph_receives_the_fields_of_its_input_type(input_state: State, expected: State) -> None:
    received: list[State] = []
    child = StateGraph(State, input_schema=TopicsInput)
    child.add_node("receive", received.append)
    child.add_edge(Endpoint.start, "receive")
    builder = StateGraph(State)
    builder.add_node("child", child)
    builder.add_edge(Endpoint.start, "child")

    asyncio.run(builder.compile().ainvoke(input_state))

    assert received == [expected], f"Expected the child to receive {expected}, got {received}."


@pytest.mark.parametrize("visits", [[], ["input"], ["input", "other"]], ids=repr)
def test_child_graph_writes_the_fields_of_its_output_type(visits: list[str]) -> None:
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
    def provide_name() -> str:
        return "real"

    def write_name(state: State, name: Annotated[str, Depends(provide_name)]) -> JokesOutput:
        return {"jokes": [name]}

    builder = nest_graph(build_reader_graph(write_name, "write"), depth)

    actual = asyncio.run(builder.compile(dependency_overrides={provide_name: lambda: replacement}).ainvoke({}))

    assert actual.get("jokes", []) == [replacement], f"Expected the child to receive {replacement!r}, got {actual}."


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_dependencies_of_a_graph_include_those_of_its_child_graphs(depth: int) -> None:
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
    names = [f"node-{index}" for index in range(child_steps)]
    graph = nest_graph(build_chain_graph(names), depth).compile()

    with expectation:
        asyncio.run(graph.ainvoke({}, step_limit=step_limit))


@pytest.mark.parametrize("name", ["child", "other"])
def test_child_graph_with_the_name_of_a_node_is_rejected(name: str) -> None:
    builder = StateGraph(State)
    builder.add_node(name, lambda state: None)

    with pytest.raises(ValueError):
        builder.add_node(name, build_chain_graph(["first"]))

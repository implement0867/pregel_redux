import asyncio
from collections.abc import Callable
from typing import Annotated, Literal, TypedDict

import pytest

from pregel_redux import Command, Endpoint, Reducer, StateGraph
from tests.conftest import join_lists


class State(TypedDict, total=False):
    visits: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class Output(TypedDict):
    visits: list[str]


def build_visitor(name: str) -> Callable[[State], State]:
    def visit(state: State) -> State:
        return {"visits": [name]}

    return visit


def build_branches(long_branch: int) -> StateGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("short", build_visitor("short"))
    builder.add_node("target", build_visitor("target"))
    builder.add_edge(Endpoint.start, "short")
    previous: str | Literal[Endpoint.start] = Endpoint.start
    for index in range(long_branch):
        builder.add_node(f"long {index}", build_visitor(f"long {index}"))
        builder.add_edge(previous, f"long {index}")
        previous = f"long {index}"
    return builder


@pytest.mark.parametrize("long_branch", [1, 2, 4])
def test_join_starts_its_target_one_time_after_all_sources_finished(long_branch: int) -> None:
    """
    Arrange: two branches of different lengths, and a join of their last nodes.
    Act: run the graph.
    Assert: the target runs one time, after all other nodes.
    """
    builder = build_branches(long_branch)
    builder.add_edge(("short", f"long {long_branch - 1}"), "target")

    actual = asyncio.run(builder.compile().ainvoke({}))["visits"]

    assert actual.count("target") == 1, f"Expected one visit of the target, got {actual}."
    assert actual[-1] == "target", f"Expected the target after all sources, got {actual}."
    assert len(actual) == long_branch + 2, f"Expected each source to run one time, got {actual}."


@pytest.mark.parametrize("long_branch", [2, 4])
def test_separate_edges_start_their_target_in_each_step_that_a_source_finished(long_branch: int) -> None:
    """
    Arrange: two branches of different lengths, and a separate edge from each last node to the target.
    Act: run the graph.
    Assert: the target runs two times.
    """
    builder = build_branches(long_branch)
    builder.add_edge("short", "target")
    builder.add_edge(f"long {long_branch - 1}", "target")

    actual = asyncio.run(builder.compile().ainvoke({}))["visits"]

    assert actual.count("target") == 2, f"Expected two visits of the target, got {actual}."


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_join_waits_for_all_sources_again_after_its_target_ran(rounds: int) -> None:
    """
    Arrange: a join of two nodes, and a target that starts both nodes again for a number of rounds.
    Act: run the graph.
    Assert: the target runs one time in each round.
    """

    def start_next_round(state: State) -> Command[Literal["first", "second"]]:
        finished = state.get("visits", []).count("target") + 1 == rounds
        return Command(update={"visits": ["target"]}, goto=() if finished else ("first", "second"))

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("first", build_visitor("first"))
    builder.add_node("second", build_visitor("second"))
    builder.add_node("target", start_next_round)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start, "second")
    builder.add_edge(("first", "second"), "target")

    actual = asyncio.run(builder.compile().ainvoke({}, step_limit=rounds * 2))["visits"]

    expected = ["first", "second", "target"] * rounds
    assert actual == expected, f"Expected {expected}, got {actual}. Each round must need both sources again."


def test_join_does_not_start_its_target_while_a_source_has_not_finished() -> None:
    """
    Arrange: a join of two nodes, of which the graph starts only one.
    Act: run the graph.
    Assert: the target does not run.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("first", build_visitor("first"))
    builder.add_node("second", build_visitor("second"))
    builder.add_node("target", build_visitor("target"))
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(("first", "second"), "target")

    actual = asyncio.run(builder.compile().ainvoke({}))["visits"]

    expected = ["first"]
    assert actual == expected, f"Expected {expected}, got {actual}. The join must wait for the second source."


def test_join_to_the_end_of_the_graph_starts_no_node() -> None:
    """
    Arrange: a join of two nodes with Endpoint.end as its target.
    Act: run the graph.
    Assert: each source runs one time.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("first", build_visitor("first"))
    builder.add_node("second", build_visitor("second"))
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start, "second")
    builder.add_edge(("first", "second"), Endpoint.end)

    actual = asyncio.run(builder.compile().ainvoke({}))["visits"]

    expected = ["first", "second"]
    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize(
    ("sources", "target"),
    [(("first", "missing"), "second"), (("first", "second"), "missing")],
)
def test_join_of_an_unknown_node_is_rejected_by_compile(sources: tuple[str, str], target: str) -> None:
    """
    Arrange: a join that names a node that is not in the graph.
    Act: compile the graph.
    Assert: compile raises a ValueError that names the node.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("first", build_visitor("first"))
    builder.add_node("second", build_visitor("second"))
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(sources, target)

    with pytest.raises(ValueError, match="missing"):
        builder.compile()


@pytest.mark.parametrize("long_branch", [2, 4])
def test_deferred_node_runs_one_time_after_no_other_node_can_run(long_branch: int) -> None:
    """
    Arrange: two branches of different lengths, with a separate edge from each last node to a deferred target.
    Act: run the graph.
    Assert: the target runs one time, after all other nodes.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("short", build_visitor("short"))
    builder.add_node("target", build_visitor("target"), defer=True)
    builder.add_edge(Endpoint.start, "short")
    builder.add_edge("short", "target")
    previous: str | Literal[Endpoint.start] = Endpoint.start
    for index in range(long_branch):
        builder.add_node(f"long {index}", build_visitor(f"long {index}"))
        builder.add_edge(previous, f"long {index}")
        previous = f"long {index}"
    builder.add_edge(previous, "target")

    actual = asyncio.run(builder.compile().ainvoke({}))["visits"]

    assert actual.count("target") == 1, f"Expected one visit of the target, got {actual}."
    assert actual[-1] == "target", f"Expected the target after all other nodes, got {actual}."


@pytest.mark.parametrize("extra_steps", [1, 3])
def test_join_to_a_deferred_node_waits_until_no_other_node_can_run(extra_steps: int) -> None:
    """
    Arrange: a join of two nodes to a deferred target, and a chain of nodes that still runs after the join is complete.
    Act: run the graph.
    Assert: the target runs one time, after the chain.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("first", build_visitor("first"))
    builder.add_node("second", build_visitor("second"))
    builder.add_node("target", build_visitor("target"), defer=True)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start, "second")
    builder.add_edge(("first", "second"), "target")
    previous = "second"
    for index in range(extra_steps):
        builder.add_node(f"extra {index}", build_visitor(f"extra {index}"))
        builder.add_edge(previous, f"extra {index}")
        previous = f"extra {index}"

    actual = asyncio.run(builder.compile().ainvoke({}))["visits"]

    expected = ["first", "second", *(f"extra {index}" for index in range(extra_steps)), "target"]
    assert actual == expected, f"Expected {expected}, got {actual}."

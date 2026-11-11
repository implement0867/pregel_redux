import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, TypedDict

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, Field

from pregel_redux import (
    Checkpoint,
    Command,
    CompiledGraph,
    Depends,
    Endpoint,
    Reducer,
    Scope,
    Send,
    StateGraph,
)
from pregel_redux.checkpoints import PendingSend
from tests.conftest import Journal, join_lists


class Review(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: int
    note: str = Field(alias="comment")
    author: str = Field(default="", validation_alias="writer")
    reader: str = Field(default="", serialization_alias="audience")


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    at: datetime
    seats: tuple[int, ...]


@dataclass(frozen=True)
class Budget:
    limit: int
    spent: int = 0


class State(TypedDict, total=False):
    topics: list[str]
    jokes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]
    budget: Budget
    review: Review
    started: datetime
    tags: frozenset[str]
    weights: dict[int, float]
    event: Event


class TopicRequest(TypedDict):
    topic: str


class JokesOutput(TypedDict, total=False):
    jokes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


def plan_jokes(state: State) -> Command[str]:
    return Command(goto=[*(Send("write", {"topic": topic}) for topic in state.get("topics", [])), "spend"])


def write_joke(request: TopicRequest) -> State:
    return {"jokes": [f"a joke about {request['topic']}"]}


def spend_budget(state: State) -> State:
    budget = state.get("budget", Budget(limit=1))
    return {"budget": Budget(budget.limit, budget.spent + 1)}


def review_jokes(state: State) -> State:
    jokes = state.get("jokes", [])
    return {"review": Review(score=len(jokes), comment=" and ".join(jokes))}


def build_joke_graph(
    review: Callable[[State], object] = review_jokes, plan: Callable[[State], object] = plan_jokes
) -> StateGraph[State, State, State]:
    builder = StateGraph(State)
    builder.add_node("plan", plan)
    builder.add_node("write", write_joke, input_schema=TopicRequest)
    builder.add_node("spend", spend_budget)
    builder.add_node("review", review)
    builder.add_edge(Endpoint.start, "plan")
    builder.add_edge("spend", "review")
    return builder


def build_sync_saver(saved: list[Checkpoint]) -> Callable[[Checkpoint], object]:
    return saved.append


def build_async_saver(saved: list[Checkpoint]) -> Callable[[Checkpoint], object]:
    async def save_checkpoint(checkpoint: Checkpoint) -> None:
        await asyncio.sleep(0)
        saved.append(checkpoint)

    return save_checkpoint


def run_with_invoke(
    graph: CompiledGraph[State, State, State], input_state: State, save_checkpoint: Callable[[Checkpoint], object]
) -> State:
    return graph.invoke(input_state, save_checkpoint=save_checkpoint)


def run_with_ainvoke(
    graph: CompiledGraph[State, State, State], input_state: State, save_checkpoint: Callable[[Checkpoint], object]
) -> State:
    return asyncio.run(graph.ainvoke(input_state, save_checkpoint=save_checkpoint))


def restore_from_json(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint.model_validate_json(checkpoint.model_dump_json())


@pytest.mark.parametrize("build_saver", [build_sync_saver, build_async_saver], ids=["sync-saver", "async-saver"])
@pytest.mark.parametrize("run_graph", [run_with_invoke, run_with_ainvoke], ids=["invoke", "ainvoke"])
def test_checkpoint_is_saved_for_the_start_and_after_each_step(
    build_saver: Callable[[list[Checkpoint]], Callable[[Checkpoint], object]],
    run_graph: Callable[[CompiledGraph[State, State, State], State, Callable[[Checkpoint], object]], State],
) -> None:
    """
    Arrange: a graph of three steps, where the first step sends one payload and starts one node by name.
    Act: run the graph with a saver.
    Assert: the saver received the start and each step, with the channel values and the pending calls as JSON data.
    """
    saved: list[Checkpoint] = []
    graph = build_joke_graph().compile()
    expected = [
        Checkpoint(values={"topics": ["cats"], "jokes": []}, nodes=("plan",), sends=()),
        Checkpoint(
            values={"topics": ["cats"], "jokes": []},
            nodes=("spend",),
            sends=(PendingSend(node="write", values={"topic": "cats"}),),
        ),
        Checkpoint(
            values={"topics": ["cats"], "jokes": ["a joke about cats"], "budget": {"limit": 1, "spent": 1}},
            nodes=("review",),
            sends=(),
        ),
        Checkpoint(
            values={
                "topics": ["cats"],
                "jokes": ["a joke about cats"],
                "budget": {"limit": 1, "spent": 1},
                "review": {"score": 1, "note": "a joke about cats", "author": "", "reader": ""},
            },
            nodes=(),
            sends=(),
        ),
    ]

    run_graph(graph, {"topics": ["cats"]}, build_saver(saved))

    assert saved == expected, f"Expected the checkpoints {expected}, got {saved}."


@settings(max_examples=40, deadline=None)
@given(
    input_state=st.fixed_dictionaries(
        {},
        optional={
            "topics": st.lists(st.text(), max_size=3),
            "jokes": st.lists(st.text(), max_size=2),
            "budget": st.builds(Budget, st.integers(), st.integers()),
            "review": st.builds(Review, score=st.integers(), comment=st.text(), writer=st.text(), reader=st.text()),
            "event": st.builds(Event, at=st.datetimes(), seats=st.lists(st.integers(), max_size=2).map(tuple)),
            "started": st.datetimes(),
            "tags": st.frozensets(st.text(), max_size=3),
            "weights": st.dictionaries(st.integers(), st.floats(allow_nan=False), max_size=3),
        },
    )
)
def test_resume_from_each_checkpoint_gives_the_result_of_the_run_that_was_not_stopped(
    input_state: State,
) -> None:
    """
    Arrange: a graph with channels of several types, an input that Hypothesis generates, and the checkpoints of one run.
    Act: write each checkpoint as JSON text, read it, and resume from it.
    Assert: each resumed run gives the output of the first run and saves the same pending calls after each step.
    """
    graph = build_joke_graph().compile()
    saved: list[Checkpoint] = []
    expected = graph.invoke(input_state, save_checkpoint=saved.append)

    for index, checkpoint in enumerate(saved):
        later: list[Checkpoint] = []

        actual = graph.resume(restore_from_json(checkpoint), save_checkpoint=later.append)

        assert actual == expected, f"Expected {expected} after a resume from checkpoint {index}, got {actual}."
        pending = [(item.nodes, item.sends) for item in later]
        expected_pending = [(item.nodes, item.sends) for item in saved[index + 1 :]]
        assert pending == expected_pending, f"Expected the pending calls {expected_pending}, got {pending}."


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, ["plan", "spend", "review"]),
        (1, ["spend", "review"]),
        (2, ["review"]),
        (3, []),
    ],
    ids=["start", "after-plan", "after-spend", "finished"],
)
def test_resume_runs_only_the_nodes_after_the_checkpoint(index: int, expected: list[str]) -> None:
    """
    Arrange: a graph whose nodes record their names, and the checkpoints of one run.
    Act: resume from one checkpoint.
    Assert: only the nodes after that checkpoint ran, and the output equals the output of the first run.
    """
    visited: list[str] = []

    def record_plan(state: State) -> Command[str]:
        visited.append("plan")
        return Command(goto="spend")

    def record_spend(state: State) -> State:
        visited.append("spend")
        return spend_budget(state)

    def record_review(state: State) -> State:
        visited.append("review")
        return review_jokes(state)

    builder = StateGraph(State)
    builder.add_node("plan", record_plan)
    builder.add_node("spend", record_spend)
    builder.add_node("review", record_review)
    builder.add_edge(Endpoint.start, "plan")
    builder.add_edge("spend", "review")
    graph = builder.compile()
    saved: list[Checkpoint] = []
    first_output = graph.invoke({"topics": ["cats"]}, save_checkpoint=saved.append)
    visited.clear()

    actual = graph.resume(saved[index])

    assert visited == expected, f"Expected the nodes {expected} to run, got {visited}."
    assert actual == first_output, f"Expected the output {first_output}, got {actual}."


@pytest.mark.parametrize("topics", [[], ["cats"], ["cats", "dogs"]], ids=["no-send", "one-send", "two-sends"])
def test_run_that_failed_resumes_from_its_last_checkpoint(topics: list[str]) -> None:
    """
    Arrange: a graph whose last node raises on its first call, and the checkpoints of the run that failed.
    Act: resume from the last checkpoint.
    Assert: the output equals the output of a run without a failure, and the earlier nodes did not run again.
    """
    visited: list[str] = []
    failures = [RuntimeError("the review failed")]

    def record_plan(state: State) -> Command[str]:
        visited.append("plan")
        return plan_jokes(state)

    def fail_first_review(state: State) -> State:
        visited.append("review")
        if failures:
            raise failures.pop()
        return review_jokes(state)

    graph = build_joke_graph(review=fail_first_review, plan=record_plan).compile()
    expected = build_joke_graph().compile().invoke({"topics": topics})
    saved: list[Checkpoint] = []
    with pytest.raises(RuntimeError, match="the review failed"):
        graph.invoke({"topics": topics}, save_checkpoint=saved.append)

    actual = graph.resume(saved[-1], save_checkpoint=saved.append)

    assert actual == expected, f"Expected the output {expected}, got {actual}."
    assert visited == ["plan", "review", "review"], f"Expected the plan to run one time, got {visited}."
    assert saved[-1].nodes == (), f"Expected the resumed run to save a finished checkpoint, got {saved[-1]}."


@pytest.mark.parametrize(
    "cleanup_error", [None, OSError("the close failed")], ids=["cleanup-succeeds", "cleanup-raises"]
)
def test_checkpoint_of_a_step_is_saved_before_the_cleanup_of_its_nodes(cleanup_error: OSError | None) -> None:
    """
    Arrange: a node with a node-scoped dependency whose cleanup records an event and can raise.
    Act: run the graph with a saver that records an event.
    Assert: the checkpoint of the step was saved before the cleanup, also when the cleanup raises.
    """
    events: list[str] = []

    def open_session() -> Iterator[str]:
        yield "session"
        events.append("close")
        if cleanup_error is not None:
            raise cleanup_error

    def use_session(state: State, session: Annotated[str, Depends(open_session)]) -> State:
        return {"jokes": [session]}

    builder = StateGraph(State)
    builder.add_node("use", use_session)
    builder.add_edge(Endpoint.start, "use")
    saved: list[Checkpoint] = []
    expected_events = ["save", "save", "close"]

    def save_checkpoint(checkpoint: Checkpoint) -> None:
        events.append("save")
        saved.append(checkpoint)

    if cleanup_error is None:
        builder.compile().invoke({}, save_checkpoint=save_checkpoint)
    else:
        with pytest.raises(OSError, match="the close failed"):
            builder.compile().invoke({}, save_checkpoint=save_checkpoint)

    assert events == expected_events, f"Expected the events {expected_events}, got {events}."
    assert saved[-1].values == {"jokes": ["session"]}, f"Expected the write of the node in {saved[-1]}."


@pytest.mark.parametrize("child_steps", [1, 2, 3], ids=["one-child-step", "two-child-steps", "three-child-steps"])
def test_child_graph_is_one_step_in_the_checkpoints_of_its_parent(child_steps: int) -> None:
    """
    Arrange: a parent graph with one child graph of several steps and one node after it.
    Act: run the parent graph with a saver.
    Assert: the saver received the start, the child graph as one step, and the last node.
    """
    child = StateGraph(State, output_schema=JokesOutput)
    for index in range(child_steps):
        child.add_node(f"write_{index}", lambda state: {"jokes": ["a joke"]})
        child.add_edge(Endpoint.start if index == 0 else f"write_{index - 1}", f"write_{index}")
    builder = StateGraph(State)
    builder.add_node("child", child)
    builder.add_node("review", review_jokes)
    builder.add_edge(Endpoint.start, "child")
    builder.add_edge("child", "review")
    saved: list[Checkpoint] = []
    expected = [("child",), ("review",), ()]

    builder.compile().invoke({}, save_checkpoint=saved.append)

    actual = [checkpoint.nodes for checkpoint in saved]
    assert actual == expected, f"Expected the pending nodes {expected}, got {actual}."
    assert saved[1].values["jokes"] == ["a joke"] * child_steps, f"Expected the output of the child in {saved[1]}."


@pytest.mark.parametrize(
    "checkpoint",
    [
        Checkpoint(values={}, nodes=("missing",), sends=()),
        Checkpoint(values={}, nodes=("plan", "missing"), sends=()),
        Checkpoint(values={}, nodes=(), sends=(PendingSend(node="missing", values={"topic": "cats"}),)),
        Checkpoint(values={"missing": 1}, nodes=("plan",), sends=()),
        Checkpoint(values={"topics": "cats"}, nodes=("plan",), sends=()),
        Checkpoint(values={"budget": {"spent": 1}}, nodes=("plan",), sends=()),
        Checkpoint(values={}, nodes=(), sends=(PendingSend(node="write", values={"topic": ["cats"]}),)),
        Checkpoint(values={}, nodes=(), sends=(PendingSend(node="write", values={"missing": "cats"}),)),
    ],
    ids=[
        "unknown-node",
        "known-and-unknown-node",
        "send-to-an-unknown-node",
        "unknown-channel",
        "value-of-the-wrong-type",
        "value-without-a-required-field",
        "send-value-of-the-wrong-type",
        "send-value-for-an-unknown-channel",
    ],
)
def test_checkpoint_that_does_not_fit_the_graph_is_rejected(checkpoint: Checkpoint) -> None:
    """
    Arrange: a checkpoint with a node, a channel or a value that the graph does not have or cannot hold.
    Act: resume from the checkpoint.
    Assert: a ValueError is raised and no node runs.
    """
    visited: list[str] = []

    def record_plan(state: State) -> Command[str]:
        visited.append("plan")
        return Command()

    graph = build_joke_graph(plan=record_plan).compile()

    with pytest.raises(ValueError):
        graph.resume(checkpoint)

    assert visited == [], f"Expected no node to run, got {visited}."


@pytest.mark.parametrize("topics", [[], ["cats"]], ids=["no-topic", "one-topic"])
def test_checkpoint_without_a_reducer_channel_resumes_with_the_starting_value_of_the_reducer(
    topics: list[str],
) -> None:
    """
    Arrange: a checkpoint that has no value for a reducer channel, as when the channel is newer than the checkpoint.
    Act: resume from the checkpoint.
    Assert: the reducer channel started from its starting value.
    """
    graph = build_joke_graph().compile()
    checkpoint = Checkpoint(values={"topics": [*topics]}, nodes=("plan",), sends=())
    expected = [f"a joke about {topic}" for topic in topics]

    actual = graph.resume(checkpoint)

    assert actual.get("jokes") == expected, f"Expected the jokes {expected}, got {actual}."


@pytest.mark.parametrize(
    ("first_limit", "resume_limit"), [(1, 2), (2, 1), (2, 25)], ids=["one-then-two", "two-then-one", "two-then-default"]
)
def test_resume_counts_its_steps_from_zero(first_limit: int, resume_limit: int) -> None:
    """
    Arrange: a graph of three steps, and the checkpoints of a run that reached its step limit.
    Act: resume from the last checkpoint with a step limit that is enough for the steps that remain.
    Assert: the resumed run gives the output of a run without a step limit failure.
    """
    graph = build_joke_graph().compile()
    expected = graph.invoke({"topics": ["cats"]})
    saved: list[Checkpoint] = []
    with pytest.raises(RuntimeError):
        graph.invoke({"topics": ["cats"]}, step_limit=first_limit, save_checkpoint=saved.append)

    actual = graph.resume(saved[-1], step_limit=resume_limit)

    assert actual == expected, f"Expected the output {expected}, got {actual}."


@pytest.mark.parametrize("failing_save", [1, 2, 3], ids=["start", "first-step", "second-step"])
def test_exception_of_the_saver_leaves_the_run(failing_save: int) -> None:
    """
    Arrange: a saver that raises on one of its calls.
    Act: run the graph with the saver.
    Assert: the exception leaves the run, and no later step runs.
    """
    saved: list[Checkpoint] = []

    def save_checkpoint(checkpoint: Checkpoint) -> None:
        saved.append(checkpoint)
        if len(saved) == failing_save:
            raise OSError("the disk is full")

    with pytest.raises(OSError, match="the disk is full"):
        build_joke_graph().compile().invoke({"topics": ["cats"]}, save_checkpoint=save_checkpoint)

    assert len(saved) == failing_save, f"Expected {failing_save} calls of the saver, got {len(saved)}."


@pytest.mark.parametrize("index", [0, 1], ids=["start", "after-the-first-step"])
def test_resume_opens_the_run_scoped_dependencies_again(journal: Journal, index: int) -> None:
    """
    Arrange: a graph of two nodes that use one run-scoped dependency, and the checkpoints of one run.
    Act: resume from a checkpoint.
    Assert: the resumed run opened the dependency one more time.
    """

    def open_client() -> int:
        return journal.open_resource("client")

    def use_client(state: State, client: Annotated[int, Depends(open_client, scope=Scope.run)]) -> State:
        return {"jokes": [f"client {client}"]}

    builder = StateGraph(State)
    builder.add_node("first", use_client)
    builder.add_node("second", use_client)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge("first", "second")
    graph = builder.compile()
    saved: list[Checkpoint] = []
    graph.invoke({}, save_checkpoint=saved.append)

    actual = graph.resume(saved[index])

    assert journal.opened == 2, f"Expected the dependency to open two times, got {journal.opened}."
    assert actual.get("jokes", [])[-1] == "client 2", f"Expected the last node to use the new value, got {actual}."


def test_resume_rejects_a_graph_that_has_an_async_node() -> None:
    """
    Arrange: a graph with one async node, and a checkpoint of it.
    Act: call the sync resume.
    Assert: a TypeError is raised, and the async resume gives the output.
    """

    async def write_joke_later(state: State) -> State:
        await asyncio.sleep(0)
        return {"jokes": ["a joke"]}

    builder = StateGraph(State)
    builder.add_node("write", write_joke_later)
    builder.add_edge(Endpoint.start, "write")
    graph = builder.compile()
    checkpoint = Checkpoint(values={}, nodes=("write",), sends=())

    with pytest.raises(TypeError):
        graph.resume(checkpoint)

    actual = asyncio.run(graph.aresume(checkpoint))
    assert actual.get("jokes") == ["a joke"], f"Expected the async resume to run the node, got {actual}."

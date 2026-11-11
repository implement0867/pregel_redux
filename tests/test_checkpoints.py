import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, TypedDict
from uuid import UUID

import pytest
from pydantic import BaseModel, ValidationError

from pregel_redux import (
    Checkpoint,
    Checkpointer,
    Command,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupted,
    PersistentGraph,
    Reducer,
    Send,
    StateGraph,
)
from pregel_redux.checkpoints import CheckpointTuple, PendingWrite
from tests.conftest import join_lists


class State(TypedDict, total=False):
    topics: list[str]
    jokes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class NarrowState(TypedDict, total=False):
    jokes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class Output(TypedDict):
    jokes: list[str]


class DictRequest(TypedDict):
    topic: str


@dataclass
class DataclassRequest:
    topic: str


class ModelRequest(BaseModel):
    topic: str


def read_dict_topic(request: DictRequest) -> str:
    return request["topic"]


def read_attribute_topic(request: DataclassRequest | ModelRequest) -> str:
    return request.topic


def plan_jokes(state: State) -> Command[str]:
    return Command(goto=[Send("write", {"topic": topic}) for topic in state.get("topics", [])])


def build_failing_writer(name: str, failures: list[Exception], calls: list[str]) -> Callable[[object], State]:
    def write(state: object) -> State:
        calls.append(name)
        if failures:
            raise failures.pop()
        return {"jokes": [name]}

    return write


def build_joke_graph(
    checkpointer: InMemoryCheckpointer,
    review_failures: list[Exception],
    calls: list[str],
    request_type: type[object] = DictRequest,
    read_topic: Callable[..., str] = read_dict_topic,
) -> PersistentGraph[State, State, Output]:
    def write_joke(request: object) -> State:
        calls.append("write")
        return {"jokes": [f"joke about {read_topic(request)}"]}

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("plan", plan_jokes)
    builder.add_node("write", write_joke, input_schema=request_type)
    builder.add_node("review", build_failing_writer("review", review_failures, calls))
    builder.add_edge(Endpoint.start, "plan")
    builder.add_edge("write", "review")
    return builder.compile(checkpointer=checkpointer)


def build_join_graph(
    checkpointer: InMemoryCheckpointer, failures: list[Exception], calls: list[str], *, with_join: bool
) -> PersistentGraph[NarrowState, NarrowState, Output]:
    builder = StateGraph(NarrowState, output_schema=Output)
    for name in ("first", "second", "target"):
        builder.add_node(name, build_failing_writer(name, [], calls))
    builder.add_node("last", build_failing_writer("last", failures, calls))
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start, "second")
    builder.add_edge("second", "last")
    if with_join:
        builder.add_edge(("first", "last"), "target")
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.parametrize(("topics", "steps"), [([], 1), (["cats"], 3), (["cats", "dogs"], 3)])
def test_thread_holds_a_checkpoint_for_the_input_and_for_each_step(topics: list[str], steps: int) -> None:
    """
    Arrange: a graph with a checkpointer.
    Act: run the graph on a thread.
    Assert: the thread holds one checkpoint for the input, one for the start node and one for each step.
    """
    checkpointer = InMemoryCheckpointer()
    graph = build_joke_graph(checkpointer, [], [])

    asyncio.run(graph.ainvoke({"topics": topics}, thread_id="thread"))

    actual = len(checkpointer.threads["thread"][""])
    assert actual == steps + 2, f"Expected {steps + 2} checkpoints, got {actual}."


@pytest.mark.parametrize(
    ("request_type", "read_topic"),
    [(DictRequest, read_dict_topic), (DataclassRequest, read_attribute_topic), (ModelRequest, read_attribute_topic)],
)
@pytest.mark.parametrize("topics", [["cats"], ["cats", "dogs"]])
def test_resume_of_a_failed_run_gives_the_output_of_a_run_that_did_not_fail(
    request_type: type[object], read_topic: Callable[..., str], topics: list[str]
) -> None:
    """
    Arrange: a run that failed in the step after its sends, and a run of the same graph that did not fail.
    Act: resume the thread of the failed run.
    Assert: the resume gives the output of the run that did not fail.
    """
    expected = asyncio.run(
        build_joke_graph(InMemoryCheckpointer(), [], [], request_type, read_topic).ainvoke(
            {"topics": topics}, thread_id="thread"
        )
    )
    graph = build_joke_graph(InMemoryCheckpointer(), [ConnectionError("no network")], [], request_type, read_topic)
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({"topics": topics}, thread_id="thread"))

    actual = asyncio.run(graph.ainvoke(None, thread_id="thread"))

    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize("topics", [["cats"], ["cats", "dogs"]])
def test_resume_starts_at_the_step_that_failed(topics: list[str]) -> None:
    """
    Arrange: a run that failed in its last step.
    Act: resume the thread.
    Assert: only the node of the last step runs again.
    """
    calls: list[str] = []
    graph = build_joke_graph(InMemoryCheckpointer(), [ConnectionError("no network")], calls)
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({"topics": topics}, thread_id="thread"))
    calls.clear()

    asyncio.run(graph.ainvoke(None, thread_id="thread"))

    expected = ["review"]
    assert calls == expected, f"Expected the calls {expected}, got {calls}."


def test_resume_of_a_pending_send_starts_its_node_with_the_stored_payload() -> None:
    """
    Arrange: a thread whose latest checkpoint is the one after the step that made the sends.
    Act: resume the thread.
    Assert: the output has one joke for each stored payload.
    """
    checkpointer = InMemoryCheckpointer()
    graph = build_joke_graph(checkpointer, [], [])
    asyncio.run(graph.ainvoke({"topics": ["cats", "dogs"]}, thread_id="thread"))
    checkpointer.threads["thread"][""] = checkpointer.threads["thread"][""][:3]

    actual = asyncio.run(graph.ainvoke(None, thread_id="thread"))

    expected = Finished({"jokes": ["joke about cats", "joke about dogs", "review"]})
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_resume_keeps_the_sources_of_a_join_that_finished_before_the_failure() -> None:
    """
    Arrange: a join whose first source finished, and a run that failed in its second source.
    Act: resume the thread.
    Assert: only the second source and the target run.
    """
    calls: list[str] = []
    graph = build_join_graph(InMemoryCheckpointer(), [ConnectionError("no network")], calls, with_join=True)
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({}, thread_id="thread"))
    calls.clear()

    actual = asyncio.run(graph.ainvoke(None, thread_id="thread"))

    expected = Finished({"jokes": ["first", "second", "last", "target"]})
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == ["last", "target"], f"Expected the calls ['last', 'target'], got {calls}."


@pytest.mark.parametrize(
    ("second_thread", "expected"),
    [("thread", ["joke about cats", "review", "joke about dogs", "review"]), ("other", ["joke about dogs", "review"])],
)
def test_run_on_a_thread_starts_with_the_values_of_its_latest_checkpoint(
    second_thread: str, expected: list[str]
) -> None:
    """
    Arrange: a thread that holds the checkpoints of a finished run.
    Act: run the graph again on that thread or on a different thread.
    Assert: only the run on the same thread adds to the stored values.
    """
    graph = build_joke_graph(InMemoryCheckpointer(), [], [])
    asyncio.run(graph.ainvoke({"topics": ["cats"]}, thread_id="thread"))

    actual = asyncio.run(graph.ainvoke({"topics": ["dogs"]}, thread_id=second_thread))

    assert actual == Finished({"jokes": expected}), f"Expected {expected}, got {actual}."


def test_resume_of_a_thread_without_a_checkpoint_is_rejected() -> None:
    """
    Arrange: a graph with a checkpointer that holds no thread.
    Act: resume a thread.
    Assert: the resume raises a LookupError that names the thread.
    """
    graph = build_joke_graph(InMemoryCheckpointer(), [], [])

    with pytest.raises(LookupError, match="'missing'"):
        asyncio.run(graph.ainvoke(None, thread_id="missing"))


def build_graph_without_review(checkpointer: InMemoryCheckpointer) -> PersistentGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("plan", plan_jokes)
    builder.add_edge(Endpoint.start, "plan")
    return builder.compile(checkpointer=checkpointer)


def build_graph_without_topics(checkpointer: InMemoryCheckpointer) -> PersistentGraph[NarrowState, NarrowState, Output]:
    builder = StateGraph(NarrowState, output_schema=Output)
    builder.add_node("review", build_failing_writer("review", [], []))
    builder.add_edge(Endpoint.start, "review")
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.parametrize(
    ("build_other_graph", "message"),
    [(build_graph_without_review, "review"), (build_graph_without_topics, "topics")],
)
def test_checkpoint_of_a_different_graph_is_rejected(
    build_other_graph: Callable[[InMemoryCheckpointer], PersistentGraph[object, object, Output]], message: str
) -> None:
    """
    Arrange: a thread of a run that failed, and a graph that does not have a node or a channel of that run.
    Act: resume the thread with the other graph.
    Assert: the resume raises a ValueError that names the node or the channel.
    """
    checkpointer = InMemoryCheckpointer()
    graph = build_joke_graph(checkpointer, [ConnectionError("no network")], [])
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({"topics": ["cats"]}, thread_id="thread"))

    with pytest.raises(ValueError, match=message):
        asyncio.run(build_other_graph(checkpointer).ainvoke(None, thread_id="thread"))


def test_checkpoint_with_a_join_that_the_graph_does_not_have_is_rejected() -> None:
    """
    Arrange: a thread of a failed run whose checkpoint holds a source of a join, and the same graph without the join.
    Act: resume the thread with the graph without the join.
    Assert: the resume raises a ValueError that names the join.
    """
    checkpointer = InMemoryCheckpointer()
    graph = build_join_graph(checkpointer, [ConnectionError("no network")], [], with_join=True)
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({}, thread_id="thread"))

    with pytest.raises(ValueError, match="JoinKey"):
        asyncio.run(build_join_graph(checkpointer, [], [], with_join=False).ainvoke(None, thread_id="thread"))


def test_invoke_without_an_input_gives_the_output_of_ainvoke() -> None:
    """
    Arrange: a run that failed with invoke, and a run of the same graph that did not fail.
    Act: resume the thread with resume.
    Assert: the resume gives the output of the run that did not fail.
    """
    expected = asyncio.run(
        build_joke_graph(InMemoryCheckpointer(), [], []).ainvoke({"topics": ["cats"]}, thread_id="thread")
    )
    graph = build_joke_graph(InMemoryCheckpointer(), [ConnectionError("no network")], [])
    with pytest.raises(ConnectionError):
        graph.invoke({"topics": ["cats"]}, thread_id="thread")

    actual = graph.invoke(None, thread_id="thread")

    assert actual == expected, f"Expected {expected}, got {actual}."


def test_invoke_without_an_input_rejects_a_graph_that_has_an_async_node() -> None:
    """
    Arrange: a graph with a checkpointer and an async node.
    Act: call invoke without an input.
    Assert: invoke raises a TypeError that names ainvoke.
    """

    async def write_later(state: State) -> State:
        return {"jokes": ["later"]}

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("later", write_later)
    builder.add_edge(Endpoint.start, "later")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(TypeError, match="ainvoke"):
        graph.invoke(None, thread_id="thread")


def build_child_graph(failures: list[Exception], calls: list[str]) -> StateGraph[State, State, NarrowState]:
    child = StateGraph(State, output_schema=NarrowState)
    child.add_node("child first", build_failing_writer("child first", [], calls))
    child.add_node("child last", build_failing_writer("child last", failures, calls))
    child.add_edge(Endpoint.start, "child first")
    child.add_edge("child first", "child last")
    return child


@dataclass
class FailingCheckpointer:
    failing_put: int | None
    stored: InMemoryCheckpointer = field(default_factory=InMemoryCheckpointer)
    puts: int = 0

    async def aput(self, thread_id: str, checkpoint: Checkpoint) -> None:
        self.puts += 1
        if self.puts == self.failing_put:
            raise ConnectionError("no database")
        await self.stored.aput(thread_id, checkpoint)

    async def aput_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        await self.stored.aput_writes(thread_id, checkpoint_ns, checkpoint_id, write)

    async def aget_tuple(self, thread_id: str, checkpoint_ns: str) -> CheckpointTuple | None:
        return await self.stored.aget_tuple(thread_id, checkpoint_ns)


def build_parent_graph(
    child_failures: list[Exception], calls: list[str], checkpointer: Checkpointer
) -> PersistentGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("child", build_child_graph(child_failures, calls))
    builder.add_node("after", build_failing_writer("after", [], calls))
    builder.add_edge(Endpoint.start, "child")
    builder.add_edge("child", "after")
    return builder.compile(checkpointer=checkpointer)


def build_fan_out_graph(calls: list[str], checkpointer: Checkpointer) -> PersistentGraph[State, State, Output]:
    def start_children(state: State) -> Command[str]:
        return Command(goto=[Send("child", {"topics": [topic]}) for topic in state.get("topics", [])])

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("plan", start_children)
    builder.add_node("child", build_child_graph([], calls))
    builder.add_edge(Endpoint.start, "plan")
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.parametrize(
    ("child_failures", "failing_put", "expected_calls"),
    [
        ([ConnectionError("no network")], None, ["child last", "after"]),
        ([], 7, ["after"]),
    ],
    ids=["last-node-of-the-child-graph-failed", "save-after-the-child-graph-failed"],
)
def test_resume_does_not_run_the_finished_nodes_of_a_child_graph_again(
    child_failures: list[Exception], failing_put: int | None, expected_calls: list[str]
) -> None:
    """
    Arrange: a run that failed inside a child graph or in the save after the child graph, and a run that did not fail.
    Act: resume the thread.
    Assert: the output is that of the run that did not fail, and no finished node of the child graph runs again.
    """
    expected = asyncio.run(build_parent_graph([], [], InMemoryCheckpointer()).ainvoke({}, thread_id="thread"))
    calls: list[str] = []
    graph = build_parent_graph(child_failures, calls, FailingCheckpointer(failing_put))
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({}, thread_id="thread"))
    calls.clear()

    actual = asyncio.run(graph.ainvoke(None, thread_id="thread"))

    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == expected_calls, f"Expected the calls {expected_calls}, got {calls}."


@pytest.mark.parametrize("topics", [["cats"], ["cats", "dogs"], ["cats", "dogs", "birds"]])
def test_resume_finds_the_run_of_a_child_graph_for_each_send(topics: list[str]) -> None:
    """
    Arrange: one finished run of a child graph for each send, and a failure of the save after those runs.
    Act: resume the thread.
    Assert: the output is that of a run that did not fail, and no node of a child graph runs again.
    """
    expected = asyncio.run(build_fan_out_graph([], InMemoryCheckpointer()).ainvoke({"topics": topics}, thread_id="t"))
    calls: list[str] = []
    graph = build_fan_out_graph(calls, FailingCheckpointer(4 + 4 * len(topics)))
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({"topics": topics}, thread_id="t"))
    calls.clear()

    actual = asyncio.run(graph.ainvoke(None, thread_id="t"))

    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == [], f"Expected no calls, got {calls}."


@dataclass
class SignallingCheckpointer:
    signal_at_write: int
    signal: asyncio.Event
    stored: InMemoryCheckpointer = field(default_factory=InMemoryCheckpointer)
    writes: int = 0

    async def aput(self, thread_id: str, checkpoint: Checkpoint) -> None:
        await self.stored.aput(thread_id, checkpoint)

    async def aput_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: UUID, write: PendingWrite) -> None:
        await self.stored.aput_writes(thread_id, checkpoint_ns, checkpoint_id, write)
        self.writes += 1
        if self.writes == self.signal_at_write:
            self.signal.set()

    async def aget_tuple(self, thread_id: str, checkpoint_ns: str) -> CheckpointTuple | None:
        return await self.stored.aget_tuple(thread_id, checkpoint_ns)


@pytest.mark.parametrize("finished_nodes", [1, 2, 3])
def test_resume_does_not_run_a_node_that_finished_in_the_failed_step(finished_nodes: int) -> None:
    """
    Arrange: one step in which some nodes finished and saved their writes, and then one node failed.
    Act: run the thread again without an input.
    Assert: only the failed node runs, and the output has the writes of all nodes.
    """
    calls: list[str] = []

    async def run_thread() -> Finished[Output] | Interrupted:
        saved = asyncio.Event()
        failures: list[Exception] = [ConnectionError("no network")]

        async def fail_after_the_other_nodes(state: NarrowState) -> NarrowState:
            calls.append("failing")
            await saved.wait()
            if failures:
                raise failures.pop()
            return {"jokes": ["failing"]}

        def build_finished_node(name: str) -> Callable[[NarrowState], object]:
            async def finish(state: NarrowState) -> NarrowState:
                calls.append(name)
                return {"jokes": [name]}

            return finish

        builder = StateGraph(NarrowState, output_schema=Output)
        builder.add_node("failing", fail_after_the_other_nodes)
        builder.add_edge(Endpoint.start, "failing")
        for index in range(finished_nodes):
            builder.add_node(f"finished {index}", build_finished_node(f"finished {index}"))
            builder.add_edge(Endpoint.start, f"finished {index}")
        graph = builder.compile(checkpointer=SignallingCheckpointer(1 + finished_nodes, saved))
        with pytest.raises(ConnectionError):
            await graph.ainvoke({}, thread_id="thread")
        calls.clear()
        return await graph.ainvoke(None, thread_id="thread")

    actual = asyncio.run(run_thread())

    expected = Finished({"jokes": ["failing", *(f"finished {index}" for index in range(finished_nodes))]})
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == ["failing"], f"Expected the calls ['failing'], got {calls}."


def build_deferred_graph(
    failing: str, failures: list[Exception], calls: list[str], *, with_join: bool
) -> PersistentGraph[NarrowState, NarrowState, Output]:
    def choose_failures(name: str) -> list[Exception]:
        return failures if name == failing else []

    builder = StateGraph(NarrowState, output_schema=Output)
    for name in ("first", "second", "last"):
        builder.add_node(name, build_failing_writer(name, choose_failures(name), calls))
    builder.add_node("target", build_failing_writer("target", choose_failures("target"), calls), defer=True)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start, "second")
    builder.add_edge("second", "last")
    if with_join:
        builder.add_edge(("first", "second"), "target")
    else:
        builder.add_edge("first", "target")
    return builder.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.parametrize("with_join", [True, False], ids=["join-to-the-deferred-node", "edge-to-the-deferred-node"])
@pytest.mark.parametrize("failing", ["last", "target"])
def test_resume_keeps_the_state_of_a_deferred_node(failing: str, *, with_join: bool) -> None:
    """
    Arrange: a deferred node that waits for a longer branch, and a run that failed in that branch or in the node.
    Act: run the thread again without an input.
    Assert: the output is that of a run that did not fail, and the deferred node runs one time after the branch.
    """
    expected = asyncio.run(build_deferred_graph("none", [], [], with_join=with_join).ainvoke({}, thread_id="thread"))
    calls: list[str] = []
    graph = build_deferred_graph(failing, [ConnectionError("no network")], calls, with_join=with_join)
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({}, thread_id="thread"))
    calls.clear()

    actual = asyncio.run(graph.ainvoke(None, thread_id="thread"))

    expected_calls = ["last", "target"] if failing == "last" else ["target"]
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == expected_calls, f"Expected the calls {expected_calls}, got {calls}."


@pytest.mark.parametrize("runs", [1, 2, 3])
def test_checkpoints_of_a_thread_form_a_chain_in_the_order_of_their_ids(runs: int) -> None:
    """
    Arrange: a graph with a checkpointer.
    Act: run the graph on one thread a number of times.
    Assert: each checkpoint has format version 1, a greater id than the one before it, and that one as its parent.
    """
    checkpointer = InMemoryCheckpointer()
    graph = build_joke_graph(checkpointer, [], [])

    for _ in range(runs):
        asyncio.run(graph.ainvoke({"topics": ["cats"]}, thread_id="thread"))

    stored = [CheckpointTuple.model_validate_json(record).checkpoint for record in checkpointer.threads["thread"][""]]
    ids = [checkpoint.id for checkpoint in stored]
    parents = [checkpoint.parent_id for checkpoint in stored]
    versions = {checkpoint.v for checkpoint in stored}
    assert ids == sorted(set(ids)), f"Expected ids that increase, got {ids}."
    assert parents == [None, *ids[:-1]], f"Expected each parent to be the checkpoint before, got {parents}."
    assert versions == {1}, f"Expected format version 1, got {versions}."


def test_latest_checkpoint_is_the_one_with_the_greatest_id() -> None:
    """
    Arrange: a thread of a run that failed, whose records the storage holds in the reverse order.
    Act: run the thread again without an input.
    Assert: the run continues from the checkpoint with the greatest id, so only the node that failed runs.
    """
    expected = asyncio.run(
        build_joke_graph(InMemoryCheckpointer(), [], []).ainvoke({"topics": ["cats"]}, thread_id="thread")
    )
    checkpointer = InMemoryCheckpointer()
    calls: list[str] = []
    graph = build_joke_graph(checkpointer, [ConnectionError("no network")], calls)
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({"topics": ["cats"]}, thread_id="thread"))
    checkpointer.threads["thread"][""].reverse()
    calls.clear()

    actual = asyncio.run(graph.ainvoke(None, thread_id="thread"))

    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == ["review"], f"Expected the calls ['review'], got {calls}."


def test_checkpoint_of_another_format_version_is_rejected() -> None:
    """
    Arrange: a thread whose stored records have format version 2.
    Act: run the thread again without an input.
    Assert: the run raises the ValidationError of Pydantic.
    """
    checkpointer = InMemoryCheckpointer()
    graph = build_joke_graph(checkpointer, [], [])
    asyncio.run(graph.ainvoke({"topics": ["cats"]}, thread_id="thread"))
    records = checkpointer.threads["thread"][""]
    checkpointer.threads["thread"][""] = [record.replace('"v":1', '"v":2') for record in records]

    with pytest.raises(ValidationError):
        asyncio.run(graph.ainvoke(None, thread_id="thread"))

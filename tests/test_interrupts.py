import asyncio
from collections.abc import Callable
from typing import Annotated, TypedDict

import pytest
from pydantic import BaseModel, ValidationError

from pregel_redux import (
    Command,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupted,
    PersistentGraph,
    Reducer,
    StateGraph,
    interrupt,
)
from tests.conftest import join_lists


class State(TypedDict, total=False):
    notes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class Output(TypedDict):
    notes: list[str]


class Decision(BaseModel):
    approved: bool


def build_asking_node(name: str, questions: int, calls: list[str]) -> Callable[[State], State]:
    def ask(state: State) -> State:
        calls.append(name)
        answers = [interrupt(f"{name} question {index}", response_schema=str) for index in range(questions)]
        return {"notes": [f"{name} got {answer}" for answer in answers]}

    return ask


def build_async_asking_node(name: str, questions: int, calls: list[str]) -> Callable[[State], object]:
    async def ask(state: State) -> State:
        calls.append(name)
        answers = [interrupt(f"{name} question {index}", response_schema=str) for index in range(questions)]
        return {"notes": [f"{name} got {answer}" for answer in answers]}

    return ask


def build_writer(name: str, calls: list[str]) -> Callable[[State], State]:
    def write(state: State) -> State:
        calls.append(name)
        return {"notes": [name]}

    return write


def build_asking_graph(
    asking_nodes: int,
    questions: int,
    calls: list[str],
    build_node: Callable[[str, int, list[str]], Callable[[State], object]] = build_asking_node,
) -> PersistentGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("sibling", build_writer("sibling", calls))
    builder.add_node("after", build_writer("after", calls))
    builder.add_edge(Endpoint.start, "sibling")
    for index in range(asking_nodes):
        builder.add_node(f"ask {index}", build_node(f"ask {index}", questions, calls))
        builder.add_edge(Endpoint.start, f"ask {index}")
        builder.add_edge(f"ask {index}", "after")
    return builder.compile(checkpointer=InMemoryCheckpointer())


def read_interrupts(result: Finished[Output] | Interrupted) -> list[tuple[str, object]]:
    match result:
        case Interrupted():
            return [(item.id, item.value) for item in result.interrupts]
        case Finished():
            return []


@pytest.mark.parametrize("build_node", [build_asking_node, build_async_asking_node], ids=["sync-node", "async-node"])
def test_interrupt_stops_the_run_and_gives_its_value(
    build_node: Callable[[str, int, list[str]], Callable[[State], object]],
) -> None:
    """
    Arrange: a graph in which one node calls interrupt.
    Act: run the graph on a thread.
    Assert: the result holds the value of the interrupt, and the node after the interrupt did not run.
    """
    calls: list[str] = []
    graph = build_asking_graph(1, 1, calls, build_node)

    actual = asyncio.run(graph.ainvoke({}, thread_id="thread"))

    values = [value for _, value in read_interrupts(actual)]
    assert values == ["ask 0 question 0"], f"Expected the value of one interrupt, got {actual}."
    assert "after" not in calls, f"Expected no call of the node after the interrupt, got {calls}."


@pytest.mark.parametrize("build_node", [build_asking_node, build_async_asking_node], ids=["sync-node", "async-node"])
def test_resume_gives_the_answer_to_the_interrupt_and_runs_only_the_stopped_node_again(
    build_node: Callable[[str, int, list[str]], Callable[[State], object]],
) -> None:
    """
    Arrange: a thread that stopped at an interrupt, in a step in which another node finished.
    Act: run the thread with a Command that has a resume value.
    Assert: the interrupt returns the value, and the node that finished does not run again.
    """
    calls: list[str] = []
    graph = build_asking_graph(1, 1, calls, build_node)
    asyncio.run(graph.ainvoke({}, thread_id="thread"))
    calls.clear()

    actual = asyncio.run(graph.ainvoke(Command(resume="yes"), thread_id="thread"))

    expected = Finished({"notes": ["ask 0 got yes", "sibling", "after"]})
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert calls == ["ask 0", "after"], f"Expected the calls ['ask 0', 'after'], got {calls}."


@pytest.mark.parametrize("questions", [2, 3])
def test_node_with_more_than_one_interrupt_stops_at_each_of_them(questions: int) -> None:
    """
    Arrange: a node that calls interrupt more than one time.
    Act: run the thread, and give one resume value at a time.
    Assert: each run stops at the next interrupt, and the last run gives all answers to the node.
    """
    graph = build_asking_graph(1, questions, [])
    results = [asyncio.run(graph.ainvoke({}, thread_id="thread"))]

    results.extend(
        asyncio.run(graph.ainvoke(Command(resume=f"answer {index}"), thread_id="thread")) for index in range(questions)
    )

    stops = [[value for _, value in read_interrupts(result)] for result in results]
    expected_stops = [*([f"ask 0 question {index}"] for index in range(questions)), []]
    expected = Finished({"notes": [*(f"ask 0 got answer {index}" for index in range(questions)), "sibling", "after"]})
    assert stops == expected_stops, f"Expected the stops {expected_stops}, got {stops}."
    assert results[-1] == expected, f"Expected {expected}, got {results[-1]}."


@pytest.mark.parametrize("asking_nodes", [2, 3])
def test_resume_by_interrupt_id_answers_the_interrupts_of_one_step(asking_nodes: int) -> None:
    """
    Arrange: a thread that stopped at one interrupt in each of several nodes of one step.
    Act: run the thread with a Command that has one resume value for each interrupt id.
    Assert: each node receives the value for its interrupt.
    """
    graph = build_asking_graph(asking_nodes, 1, [])
    stopped = read_interrupts(asyncio.run(graph.ainvoke({}, thread_id="thread")))
    answers = {interrupt_id: f"answer to {value}" for interrupt_id, value in stopped}

    actual = asyncio.run(graph.ainvoke(Command(resume_by_interrupt_id=answers), thread_id="thread"))

    notes = [f"ask {index} got answer to ask {index} question 0" for index in range(asking_nodes)]
    expected = Finished({"notes": [*notes, "sibling", "after"]})
    assert len(stopped) == asking_nodes, f"Expected {asking_nodes} interrupts, got {stopped}."
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_resume_by_interrupt_id_for_one_of_two_interrupts_leaves_the_other_waiting() -> None:
    """
    Arrange: a thread that stopped at one interrupt in each of two nodes of one step.
    Act: run the thread with a resume value for only the first interrupt id.
    Assert: the run stops again, at only the second interrupt.
    """
    graph = build_asking_graph(2, 1, [])
    stopped = read_interrupts(asyncio.run(graph.ainvoke({}, thread_id="thread")))

    actual = asyncio.run(graph.ainvoke(Command(resume_by_interrupt_id={stopped[0][0]: "answer"}), thread_id="thread"))

    assert read_interrupts(actual) == [stopped[1]], f"Expected only {stopped[1]}, got {actual}."


def test_one_resume_value_for_more_than_one_waiting_interrupt_is_rejected() -> None:
    """
    Arrange: a thread that stopped at one interrupt in each of two nodes of one step.
    Act: run the thread with a Command that has one resume value and no interrupt id.
    Assert: the run raises a ValueError that names resume_by_interrupt_id.
    """
    graph = build_asking_graph(2, 1, [])
    asyncio.run(graph.ainvoke({}, thread_id="thread"))

    with pytest.raises(ValueError, match="resume_by_interrupt_id"):
        asyncio.run(graph.ainvoke(Command(resume="answer"), thread_id="thread"))


@pytest.mark.parametrize(
    ("answer", "expected"),
    [(True, "approved"), ("no", "rejected")],
    ids=["mapping-of-the-fields", "mapping-with-text-for-a-bool"],
)
def test_interrupt_returns_the_resume_value_as_its_response_schema(expected: str, *, answer: bool | str) -> None:
    """
    Arrange: a node that calls interrupt with a Pydantic model as its response schema.
    Act: resume the thread with a mapping of the fields of the model.
    Assert: the node receives an instance of the model.
    """

    def decide(state: State) -> State:
        decision = interrupt("approve?", response_schema=Decision)
        return {"notes": ["approved" if decision.approved else "rejected"]}

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("decide", decide)
    builder.add_edge(Endpoint.start, "decide")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    asyncio.run(graph.ainvoke({}, thread_id="thread"))

    actual = asyncio.run(graph.ainvoke(Command(resume={"approved": answer}), thread_id="thread"))

    assert actual == Finished({"notes": [expected]}), f"Expected the note {expected!r}, got {actual}."


def test_resume_value_that_is_not_of_the_response_schema_fails_the_node() -> None:
    """
    Arrange: a node that calls interrupt with a Pydantic model as its response schema.
    Act: resume the thread with a value that is not a mapping of the fields of the model.
    Assert: the run raises the ValidationError of Pydantic.
    """

    def decide(state: State) -> State:
        interrupt("approve?", response_schema=Decision)
        return {}

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("decide", decide)
    builder.add_edge(Endpoint.start, "decide")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    asyncio.run(graph.ainvoke({}, thread_id="thread"))

    with pytest.raises(ValidationError):
        asyncio.run(graph.ainvoke(Command(resume="text"), thread_id="thread"))


def test_run_without_an_input_keeps_the_resume_value_of_a_run_that_failed() -> None:
    """
    Arrange: a thread that received a resume value, in a run that failed in the node after the interrupt returned.
    Act: run the thread again without an input.
    Assert: the interrupt returns the stored resume value.
    """
    failures: list[Exception] = [ConnectionError("no network")]

    def ask_then_fail(state: State) -> State:
        answer = interrupt("question", response_schema=str)
        if failures:
            raise failures.pop()
        return {"notes": [answer]}

    builder = StateGraph(State, output_schema=Output)
    builder.add_node("ask", ask_then_fail)
    builder.add_edge(Endpoint.start, "ask")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())
    asyncio.run(graph.ainvoke({}, thread_id="thread"))
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke(Command(resume="stored answer"), thread_id="thread"))

    actual = asyncio.run(graph.ainvoke(None, thread_id="thread"))

    assert actual == Finished({"notes": ["stored answer"]}), f"Expected the stored answer, got {actual}."


@pytest.mark.parametrize("depth", [1, 2])
def test_interrupt_in_a_child_graph_stops_the_run_and_resumes_inside_the_child_graph(depth: int) -> None:
    """
    Arrange: a node that calls interrupt, in a child graph at a depth, after a node of the same child graph.
    Act: run the thread, then run it with a resume value.
    Assert: the first run stops at the interrupt, and the second run does not run the earlier node again.
    """
    calls: list[str] = []
    inner = StateGraph(State)
    inner.add_node("before", build_writer("before", calls))
    inner.add_node("ask", build_asking_node("ask", 1, calls))
    inner.add_edge(Endpoint.start, "before")
    inner.add_edge("before", "ask")
    for _ in range(depth - 1):
        outer = StateGraph(State)
        outer.add_node("child", inner)
        outer.add_edge(Endpoint.start, "child")
        inner = outer
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("child", inner)
    builder.add_edge(Endpoint.start, "child")
    graph = builder.compile(checkpointer=InMemoryCheckpointer())

    stopped = asyncio.run(graph.ainvoke({}, thread_id="thread"))
    calls.clear()
    actual = asyncio.run(graph.ainvoke(Command(resume="yes"), thread_id="thread"))

    values = [value for _, value in read_interrupts(stopped)]
    assert values == ["ask question 0"], f"Expected the value of one interrupt, got {stopped}."
    assert calls == ["ask"], f"Expected the calls ['ask'], got {calls}."
    assert read_interrupts(actual) == [], f"Expected a finished run, got {actual}."


def test_interrupt_in_a_graph_without_a_checkpointer_is_rejected() -> None:
    """
    Arrange: a graph without a checkpointer, in which one node calls interrupt.
    Act: run the graph.
    Assert: the run raises a RuntimeError that names the checkpointer.
    """
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("ask", build_asking_node("ask", 1, []))
    builder.add_edge(Endpoint.start, "ask")

    with pytest.raises(RuntimeError, match="checkpointer"):
        asyncio.run(builder.compile().ainvoke({}))


def test_interrupt_outside_a_graph_is_rejected() -> None:
    """
    Arrange: no running graph.
    Act: call interrupt.
    Assert: the call raises a RuntimeError.
    """
    with pytest.raises(RuntimeError, match="inside a node"):
        interrupt("question", response_schema=str)

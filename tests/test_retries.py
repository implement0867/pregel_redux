import asyncio
from collections.abc import Callable
from typing import TypedDict

import pytest

from pregel_redux import Command, Endpoint, Finished, InMemoryCheckpointer, RetryPolicy, StateGraph, interrupt


class State(TypedDict, total=False):
    note: str


class Output(TypedDict):
    note: str


def build_failing_node(failures: list[Exception], calls: list[str]) -> Callable[[State], State]:
    def write(state: State) -> State:
        calls.append("write")
        if failures:
            raise failures.pop(0)
        return {"note": "done"}

    return write


def build_graph(node: Callable[[State], object], policy: RetryPolicy | None) -> StateGraph[State, State, Output]:
    builder = StateGraph(State, output_schema=Output)
    builder.add_node("write", node, retry_policy=policy)
    builder.add_edge(Endpoint.start, "write")
    return builder


@pytest.mark.parametrize("jitter", [False, True])
def test_node_with_a_retry_policy_runs_again_after_a_failure(*, jitter: bool) -> None:
    """
    Arrange: a node that fails two times with ConnectionError, with a policy of three attempts.
    Act: run the graph.
    Assert: the run finishes with the output of the third attempt.
    """
    calls: list[str] = []
    failures: list[Exception] = [ConnectionError("one"), ConnectionError("two")]
    policy = RetryPolicy(max_attempts=3, initial_interval=0.001, max_interval=0.001, jitter=jitter)
    graph = build_graph(build_failing_node(failures, calls), policy).compile()

    actual = graph.invoke({})

    assert actual == Output(note="done"), f"Expected the output of the third attempt, got {actual}."
    assert len(calls) == 3, f"Expected 3 attempts, got {len(calls)}."


@pytest.mark.parametrize(
    ("policy", "failure", "expected_calls"),
    [
        (None, ConnectionError("no network"), 1),
        (RetryPolicy(max_attempts=2, initial_interval=0), ConnectionError("no network"), 2),
        (RetryPolicy(max_attempts=3, initial_interval=0), ValueError("bad value"), 1),
        (RetryPolicy(max_attempts=3, initial_interval=0, retry_on=KeyError), ConnectionError("no network"), 1),
        (RetryPolicy(max_attempts=3, initial_interval=0, retry_on=(KeyError, ValueError)), ValueError("bad"), 3),
        (RetryPolicy(max_attempts=3, initial_interval=0, retry_on=lambda failure: False), ConnectionError("no"), 1),
    ],
    ids=["no-policy", "attempts-used", "default-skips-value-error", "other-type", "listed-type", "function"],
)
def test_node_failure_is_raised_when_the_policy_does_not_allow_another_attempt(
    policy: RetryPolicy | None, failure: Exception, expected_calls: int
) -> None:
    """
    Arrange: a node that always fails, with a policy that stops after the expected number of attempts.
    Act: run the graph.
    Assert: the run raises the failure of the node after the expected number of attempts.
    """
    calls: list[str] = []
    graph = build_graph(build_failing_node([failure] * 5, calls), policy).compile()

    with pytest.raises(type(failure)):
        graph.invoke({})

    assert len(calls) == expected_calls, f"Expected {expected_calls} attempts, got {len(calls)}."


def test_retry_asks_the_answered_interrupts_of_the_node_again_from_the_first() -> None:
    """
    Arrange: a node that calls interrupt and then fails one time, on a thread that stopped at the interrupt.
    Act: run the thread with a resume value.
    Assert: the second attempt gets the same answer, and the run finishes.
    """
    failures: list[Exception] = [ConnectionError("no network")]

    def ask(state: State) -> State:
        answer = interrupt("question", response_schema=str)
        if failures:
            raise failures.pop(0)
        return {"note": answer}

    policy = RetryPolicy(max_attempts=2, initial_interval=0)
    graph = build_graph(ask, policy).compile(checkpointer=InMemoryCheckpointer())
    asyncio.run(graph.ainvoke({}, thread_id="thread"))

    actual = asyncio.run(graph.ainvoke(Command(resume="answer"), thread_id="thread"))

    assert actual == Finished(Output(note="answer")), f"Expected the answer, got {actual}."

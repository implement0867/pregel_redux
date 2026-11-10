import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import nullcontext as does_not_raise
from operator import add
from typing import Annotated, TypedDict

import pytest

from pregel_redux import Depends, Endpoint, Scope, StateDep, StateGraph
from pregel_redux.dependencies import read_dependency_parameters, read_state
from pregel_redux.graph import Node
from tests.conftest import Journal


class State(TypedDict, total=False):
    seen: Annotated[list[int], add]


def build_one_node_graph(node: Node[State]) -> StateGraph[State, State, State]:
    builder = StateGraph(State)
    builder.add_node("use", node)
    builder.add_edge(Endpoint.start, "use")
    return builder


def build_two_node_graph(first: Node[State], second: Node[State], *, parallel: bool) -> StateGraph[State, State, State]:
    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(*((Endpoint.start, "second") if parallel else ("first", "second")))
    return builder


def return_value() -> int:
    return 7


async def await_value() -> int:
    return 7


def yield_value() -> Iterator[int]:
    yield 7


async def yield_awaited_value() -> AsyncIterator[int]:
    yield 7


type NumberDep = Annotated[int, Depends(return_value)]


def build_graph_with_scope_conflict() -> StateGraph[State, State, State]:
    def provide_long_value(value: Annotated[int, Depends(return_value)]) -> int:
        return value

    def use_value(state: State, value: Annotated[int, Depends(provide_long_value, scope=Scope.run)]) -> None:
        return None

    return build_one_node_graph(use_value)


def build_graph_with_plain_parameter() -> StateGraph[State, State, State]:
    def use_value(state: State, config: int) -> None:
        return None

    return build_one_node_graph(use_value)


def build_graph_with_plain_provider_parameter() -> StateGraph[State, State, State]:
    def provide_number(config: int) -> int:
        return config

    def use_value(state: State, value: Annotated[int, Depends(provide_number)]) -> None:
        return None

    return build_one_node_graph(use_value)


def build_graph_with_two_markers() -> StateGraph[State, State, State]:
    def use_value(state: State, value: Annotated[int, Depends(return_value), Depends(await_value)]) -> None:
        return None

    return build_one_node_graph(use_value)


def build_graph_with_variadic_parameter() -> StateGraph[State, State, State]:
    def use_value(state: State, *values: Annotated[int, Depends(return_value)]) -> None:
        return None

    return build_one_node_graph(use_value)


def build_graph_with_state_in_run_scope() -> StateGraph[State, State, State]:
    def count_values(state: StateDep[State]) -> int:
        return len(state.get("seen", []))

    def use_value(state: State, value: Annotated[int, Depends(count_values, scope=Scope.run)]) -> None:
        return None

    return build_one_node_graph(use_value)


def build_graph_with_run_scoped_state() -> StateGraph[State, State, State]:
    def use_value(state: State, copy: Annotated[State, Depends(read_state, scope=Scope.run)]) -> None:
        return None

    return build_one_node_graph(use_value)


def build_graph_with_dependency_cycle() -> StateGraph[State, State, State]:
    def provide_first(value: int) -> int:
        return value

    def provide_second(value: Annotated[int, Depends(provide_first)]) -> int:
        return value

    provide_first.__annotations__["value"] = Annotated[int, Depends(provide_second)]

    def use_value(state: State, value: Annotated[int, Depends(provide_first)]) -> None:
        return None

    return build_one_node_graph(use_value)


@pytest.mark.parametrize("scope", list(Scope), ids=lambda scope: scope.name)
@pytest.mark.parametrize("provider", [return_value, await_value, yield_value, yield_awaited_value])
def test_provider_of_each_kind_supplies_its_value(provider: Callable[..., object], scope: Scope) -> None:
    def use_value(state: State, value: Annotated[int, Depends(provider, scope=scope)]) -> State:
        return {"seen": [value]}

    actual = asyncio.run(build_one_node_graph(use_value).compile().ainvoke({}))

    assert actual == {"seen": [7]}, f"Expected the node to receive 7 from {provider.__name__}, got {actual}."


@pytest.mark.parametrize(
    ("parallel", "expected_events"),
    [
        (False, ["open link 1", "node 1", "close 1", "open link 2", "node 2", "close 2"]),
        (True, ["open link 1", "node 1", "open link 2", "node 2", "close 2", "close 1"]),
    ],
    ids=["sequential-closes-after-each-step", "parallel-closes-after-the-step"],
)
def test_node_scoped_provider_opens_and_closes_for_each_node_call(
    journal: Journal, *, parallel: bool, expected_events: list[str]
) -> None:
    async def open_connection() -> AsyncIterator[int]:
        number = journal.open_resource("link")
        yield number
        journal.events.append(f"close {number}")

    def use_value(state: State, number: Annotated[int, Depends(open_connection)]) -> State:
        journal.events.append(f"node {number}")
        return {"seen": [number]}

    graph = build_two_node_graph(use_value, use_value, parallel=parallel).compile()

    actual = asyncio.run(graph.ainvoke({}))

    assert actual == {"seen": [1, 2]}, f"Expected one value for each node call, got {actual}."
    assert journal.events == expected_events, f"Expected {expected_events}, got {journal.events}."


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "sequential"])
def test_run_scoped_provider_opens_one_time_and_closes_when_the_run_ends(journal: Journal, *, parallel: bool) -> None:
    def open_client() -> Iterator[int]:
        number = journal.open_resource("client")
        yield number
        journal.events.append(f"close client {number}")

    async def use_value(state: State, number: Annotated[int, Depends(open_client, scope=Scope.run)]) -> State:
        journal.events.append("node")
        return {"seen": [number]}

    graph = build_two_node_graph(use_value, use_value, parallel=parallel).compile()
    expected_events = ["open client 1", "node", "node", "close client 1"]

    first = asyncio.run(graph.ainvoke({}))

    assert first == {"seen": [1, 1]}, f"Expected the two nodes to share one value, got {first}."
    assert journal.events == expected_events, f"Expected {expected_events}, got {journal.events}."

    second = asyncio.run(graph.ainvoke({}))

    assert second == {"seen": [2, 2]}, f"Expected a new value for the second run, got {second}."


def test_run_scoped_provider_of_a_node_that_never_runs_is_not_opened(journal: Journal) -> None:
    def open_client() -> int:
        return journal.open_resource("client")

    def use_value(state: State, number: Annotated[int, Depends(open_client, scope=Scope.run)]) -> State:
        return {"seen": [number]}

    builder = StateGraph(State)
    builder.add_node("start", lambda state: None)
    builder.add_node("unused", use_value)
    builder.add_edge(Endpoint.start, "start")

    actual = asyncio.run(builder.compile().ainvoke({}))

    assert actual == {}, f"Expected no writes, got {actual}."
    assert journal.events == [], f"Expected no opened provider, got {journal.events}."


def test_provider_requested_two_times_in_one_node_call_runs_one_time(journal: Journal) -> None:
    async def provide_number() -> int:
        return journal.open_resource("number")

    def double_number(value: Annotated[int, Depends(provide_number)]) -> int:
        return value * 2

    def use_value(
        state: State, value: Annotated[int, Depends(provide_number)], twice: Annotated[int, Depends(double_number)]
    ) -> State:
        return {"seen": [value, twice]}

    builder = build_one_node_graph(use_value)

    actual = asyncio.run(builder.compile().ainvoke({}))

    assert actual == {"seen": [1, 2]}, f"Expected the provider to run one time, got {actual}."
    assert builder.dependencies == {provide_number, double_number}, (
        f"Expected the two providers of the graph, got {builder.dependencies}."
    )


@pytest.mark.parametrize("error_type", [LookupError, ValueError, RuntimeError])
def test_provider_cleanup_receives_the_exception_of_the_node(error_type: type[Exception]) -> None:
    caught: list[BaseException] = []

    def open_test_resource() -> Iterator[int]:
        try:
            yield 1
        except error_type as error:
            caught.append(error)
            raise

    def raise_error(state: State, value: Annotated[int, Depends(open_test_resource)]) -> State:
        raise error_type("no value")

    with pytest.raises(error_type) as raised:
        asyncio.run(build_one_node_graph(raise_error).compile().ainvoke({}))
    assert caught == [raised.value], f"Expected the provider to receive {raised.value!r}, got {caught}."


@pytest.mark.parametrize("scope", list(Scope), ids=lambda scope: scope.name)
@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "sequential"])
@pytest.mark.parametrize("replacement", [return_value, await_value, yield_value, yield_awaited_value])
def test_override_replaces_a_provider_for_every_node(
    replacement: Callable[..., object], scope: Scope, *, parallel: bool
) -> None:
    def provide_number() -> int:
        raise NotImplementedError

    def use_value(state: State, value: Annotated[int, Depends(provide_number, scope=scope)]) -> State:
        return {"seen": [value]}

    graph = build_two_node_graph(use_value, use_value, parallel=parallel).compile({provide_number: replacement})

    actual = asyncio.run(graph.ainvoke({}))

    assert actual == {"seen": [7, 7]}, f"Expected the two nodes to receive 7 from the override, got {actual}."


@pytest.mark.parametrize(
    ("build_graph", "error"),
    [
        (build_graph_with_scope_conflict, ValueError),
        (build_graph_with_dependency_cycle, ValueError),
        (build_graph_with_state_in_run_scope, ValueError),
        (build_graph_with_run_scoped_state, ValueError),
        (build_graph_with_plain_parameter, TypeError),
        (build_graph_with_plain_provider_parameter, TypeError),
        (build_graph_with_two_markers, TypeError),
        (build_graph_with_variadic_parameter, TypeError),
    ],
)
def test_invalid_dependency_is_rejected_by_compile(
    build_graph: Callable[[], StateGraph[State, State, State]], error: type[Exception]
) -> None:
    builder = build_graph()

    with pytest.raises(error):
        builder.compile()


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "sequential"])
def test_one_provider_requested_with_two_scopes_gives_two_values(journal: Journal, *, parallel: bool) -> None:
    def provide_number() -> int:
        return journal.open_resource("number")

    def use_run_value(state: State, value: Annotated[int, Depends(provide_number, scope=Scope.run)]) -> State:
        return {"seen": [value]}

    def use_node_value(state: State, value: Annotated[int, Depends(provide_number)]) -> State:
        return {"seen": [value]}

    graph = build_two_node_graph(use_run_value, use_node_value, parallel=parallel).compile()

    actual = asyncio.run(graph.ainvoke({}))

    assert sorted(actual.get("seen", [])) == [1, 2], f"Expected two separate values, got {actual}."


@pytest.mark.parametrize("state_parameters", [1, 2])
def test_function_with_too_few_parameters_for_the_state_is_rejected(state_parameters: int) -> None:
    with pytest.raises(TypeError):
        read_dependency_parameters(lambda: None, state_parameters=state_parameters)


@pytest.mark.parametrize(
    "cleanup_error",
    [None, LookupError("cleanup failed"), ValueError("cleanup failed")],
    ids=["cleanup-succeeds", "cleanup-raises-lookup-error", "cleanup-raises-value-error"],
)
def test_node_scoped_cleanup_runs_after_the_writes_of_the_step_are_applied(
    journal: Journal, cleanup_error: Exception | None
) -> None:
    def combine_values(current: list[int], update: list[int]) -> list[int]:
        journal.events.append("reduce")
        return current + update

    class ReducedState(TypedDict, total=False):
        seen: Annotated[list[int], combine_values]

    def open_connection() -> Iterator[int]:
        yield 1
        journal.events.append("cleanup")
        if cleanup_error is not None:
            raise cleanup_error

    def use_value(state: ReducedState, value: Annotated[int, Depends(open_connection)]) -> ReducedState:
        return {"seen": [value]}

    builder = StateGraph(ReducedState)
    builder.add_node("use", use_value)
    builder.add_edge(Endpoint.start, "use")

    expectation = does_not_raise() if cleanup_error is None else pytest.raises(type(cleanup_error))

    with expectation:
        asyncio.run(builder.compile().ainvoke({"seen": [0]}))

    assert journal.events == ["reduce", "cleanup"], (
        f"Expected the reducer call before the cleanup, got {journal.events}. "
        "The update of a node must stand before its dependencies close."
    )


@pytest.mark.parametrize("error_type", [LookupError, ValueError])
def test_cleanup_of_a_finished_node_runs_when_another_node_of_the_step_fails(
    journal: Journal, error_type: type[Exception]
) -> None:
    def open_connection() -> Iterator[int]:
        number = journal.open_resource("link")
        yield number
        journal.events.append(f"close {number}")

    def use_value(state: State, number: Annotated[int, Depends(open_connection)]) -> State:
        return {"seen": [number]}

    def raise_error(state: State) -> State:
        raise error_type("no value")

    graph = build_two_node_graph(use_value, raise_error, parallel=True).compile()
    expected_events = ["open link 1", "close 1"]

    with pytest.raises(error_type):
        asyncio.run(graph.ainvoke({}))

    assert journal.events == expected_events, (
        f"Expected {expected_events}, got {journal.events}. A finished node must not keep its dependency open."
    )


@pytest.mark.parametrize("error_type", [LookupError, ValueError])
def test_provider_that_suppresses_the_exception_of_its_node_is_rejected(error_type: type[Exception]) -> None:
    def open_test_resource() -> Iterator[int]:
        try:
            yield 1
        except error_type:
            return

    def raise_error(state: State, value: Annotated[int, Depends(open_test_resource)]) -> State:
        raise error_type("no value")

    with pytest.raises(RuntimeError):
        asyncio.run(build_one_node_graph(raise_error).compile().ainvoke({}))


@pytest.mark.parametrize(
    ("parallel", "expected"),
    [(False, [0, 1]), (True, [0, 0])],
    ids=["sequential-sees-the-earlier-write", "parallel-sees-the-same-state"],
)
def test_provider_receives_the_state_of_its_node(*, parallel: bool, expected: list[int]) -> None:
    def count_values(state: StateDep[State]) -> int:
        return len(state.get("seen", []))

    def use_value(state: State, count: Annotated[int, Depends(count_values)]) -> State:
        return {"seen": [count]}

    graph = build_two_node_graph(use_value, use_value, parallel=parallel).compile()

    actual = asyncio.run(graph.ainvoke({}))

    assert actual == {"seen": expected}, f"Expected {expected}, got {actual}. A provider must receive the node state."


@pytest.mark.parametrize("seen", [[], [4], [4, 5]], ids=repr)
def test_node_receives_the_state_through_the_state_dependency(seen: list[int]) -> None:
    def use_value(state: State, copy: StateDep[State]) -> State:
        return {"seen": [len(copy.get("seen", []))]} if copy == state else {}

    actual = asyncio.run(build_one_node_graph(use_value).compile().ainvoke({"seen": seen}))

    assert actual == {"seen": [*seen, len(seen)]}, f"Expected the state dependency to equal the state, got {actual}."


def test_type_alias_with_depends_supplies_its_value() -> None:
    def use_value(state: State, value: NumberDep) -> State:
        return {"seen": [value]}

    actual = asyncio.run(build_one_node_graph(use_value).compile().ainvoke({}))

    assert actual == {"seen": [7]}, f"Expected the alias to supply 7, got {actual}."


def test_read_state_called_outside_a_graph_raises() -> None:
    with pytest.raises(RuntimeError):
        read_state()

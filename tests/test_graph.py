import asyncio
from collections.abc import Callable, Mapping
from operator import add
from typing import Annotated, Literal, TypedDict

import pytest

from pregel_redux import Endpoint, StateGraph


class State[T](TypedDict, total=False):
    items: Annotated[list[T], add]
    latest: T
    failures: int


class Input[T](TypedDict):
    items: Annotated[list[T], add]


class Output[T](TypedDict):
    items: Annotated[list[T], add]
    latest: T


def build_sequential_graph[T](produce: Callable[[], T]) -> StateGraph[State[T], Input[T], Output[T]]:
    def collect_item(state: State[T]) -> State[T]:
        return {"items": [produce()]}

    def remember_latest(state: State[T]) -> State[T]:
        return {"latest": produce()}

    builder = StateGraph(State[T], input_schema=Input[T], output_schema=Output[T])
    builder.add_node("collect", collect_item)
    builder.add_node("remember", remember_latest)
    builder.add_edge(Endpoint.start, "collect")
    builder.add_edge("collect", "remember")
    builder.add_edge("remember", Endpoint.end)
    return builder


def build_parallel_graph[T](produce: Callable[[], T]) -> StateGraph[State[T], State[T], State[T]]:
    async def write_first_item(state: State[T]) -> State[T]:
        return {"items": [produce()]}

    async def write_second_item(state: State[T]) -> State[T]:
        return {"items": [produce()]}

    builder = StateGraph(State[T])
    builder.add_node("first", write_first_item)
    builder.add_node("second", write_second_item)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start, "second")
    return builder


def build_routed_graph(results: list[bool]) -> StateGraph[State[str], State[str], State[str]]:
    def skip_update(state: State[str]) -> None:
        return None

    def do_work(state: State[str]) -> State[str]:
        if results.pop(0):
            return {"items": ["done"]}
        return {"failures": state.get("failures", 0) + 1}

    def choose_route(state: State[str]) -> str | Literal[Endpoint.end]:
        return Endpoint.end if state.get("items") else "work"

    builder = StateGraph(State[str])
    builder.add_node("router", skip_update)
    builder.add_node("work", do_work)
    builder.add_edge(Endpoint.start, "router")
    builder.add_conditional_edges("router", choose_route)
    builder.add_edge("work", "router")
    return builder


@pytest.fixture
def parallel_builder() -> StateGraph[State[int], State[int], State[int]]:
    return build_parallel_graph(lambda: 1)


@pytest.mark.parametrize(
    ("produced", "items", "expected"),
    [
        (1, [0], {"items": [0, 1], "latest": 1}),
        ("b", ["a"], {"items": ["a", "b"], "latest": "b"}),
        (1, [], {"items": [1], "latest": 1}),
    ],
    ids=["int", "str", "no-items-in-the-input"],
)
def test_sequential_partial_writes_to_a_generic_schema_keep_all_channels(
    produced: object, items: list[object], expected: Mapping[str, object]
) -> None:
    graph = build_sequential_graph(lambda: produced).compile()

    actual = asyncio.run(graph.ainvoke({"items": items}))

    assert actual == expected, f"Expected {expected}, got {actual}. A partial write must not remove other channels."


@pytest.mark.parametrize(
    ("produced", "input_state", "expected"),
    [
        (1, {"items": [0], "latest": 0}, {"items": [0, 1, 1], "latest": 0}),
        ("b", {"items": ["a"]}, {"items": ["a", "b", "b"]}),
        (1, {}, {"items": [1, 1]}),
    ],
    ids=["int", "str", "empty-input"],
)
def test_parallel_writes_to_a_generic_schema_use_the_reducer(
    produced: object, input_state: State[object], expected: Mapping[str, object]
) -> None:
    graph = build_parallel_graph(lambda: produced).compile()

    actual = asyncio.run(graph.ainvoke(input_state))

    assert actual == expected, f"Expected {expected}, got {actual}. The reducer must combine parallel writes."


def test_parallel_writes_to_a_channel_without_reducer_are_rejected() -> None:
    def write_latest(state: State[int]) -> State[int]:
        return {"latest": 1}

    builder = StateGraph(State[int])
    builder.add_node("first", write_latest)
    builder.add_node("second", write_latest)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start, "second")

    with pytest.raises(ValueError):
        asyncio.run(builder.compile().ainvoke({}))


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([True], {"items": ["done"]}),
        ([False, True], {"items": ["done"], "failures": 1}),
        ([False, False, True], {"items": ["done"], "failures": 2}),
    ],
    ids=["no-failure", "one-failure", "two-failures"],
)
def test_router_runs_a_failed_node_again_as_a_routing_decision(
    results: list[bool], expected: Mapping[str, object]
) -> None:
    graph = build_routed_graph([*results]).compile()

    actual = asyncio.run(graph.ainvoke({}))

    assert actual == expected, f"Expected {expected}, got {actual}. The router must schedule the node until it is done."


def test_router_receives_the_writes_of_its_own_node() -> None:
    def write_latest(state: State[int]) -> State[int]:
        return {"latest": 5}

    def choose_route(state: State[int]) -> list[str | Literal[Endpoint.end]]:
        return ["record"] if state.get("latest") == 5 else [Endpoint.end]

    def record_latest(state: State[int]) -> State[int]:
        return {"items": [state["latest"]]}

    builder = StateGraph(State[int])
    builder.add_node("write", write_latest)
    builder.add_node("record", record_latest)
    builder.add_edge(Endpoint.start, "write")
    builder.add_conditional_edges("write", choose_route)
    expected = {"latest": 5, "items": [5]}

    actual = asyncio.run(builder.compile().ainvoke({}))

    assert actual == expected, f"Expected {expected}, got {actual}. The router must see the write of its own node."


@pytest.mark.parametrize("step_limit", [1, 2, 4, 9])
def test_graph_that_does_not_finish_within_the_step_limit_raises(step_limit: int) -> None:
    graph = build_routed_graph([False] * 10).compile()

    with pytest.raises(RuntimeError):
        asyncio.run(graph.ainvoke({}, step_limit=step_limit))


@pytest.mark.parametrize(
    "failure",
    [LookupError("no value"), ValueError("wrong value"), KeyError("key")],
    ids=lambda failure: type(failure).__name__,
)
def test_exception_of_one_node_leaves_the_run_unchanged(
    parallel_builder: StateGraph[State[int], State[int], State[int]], failure: Exception
) -> None:
    def raise_failure(state: State[int]) -> State[int]:
        raise failure

    parallel_builder.add_node("fail", raise_failure)
    parallel_builder.add_edge(Endpoint.start, "fail")

    with pytest.raises(type(failure)) as raised:
        asyncio.run(parallel_builder.compile().ainvoke({}))
    assert raised.value is failure, f"Expected {failure!r}, got {raised.value!r}. The exception must not be wrapped."


def test_exceptions_of_two_nodes_leave_the_run_as_a_group(
    parallel_builder: StateGraph[State[int], State[int], State[int]],
) -> None:
    def raise_lookup_error(state: State[int]) -> State[int]:
        raise LookupError("no value")

    def raise_value_error(state: State[int]) -> State[int]:
        raise ValueError("wrong value")

    parallel_builder.add_node("lookup", raise_lookup_error)
    parallel_builder.add_node("value", raise_value_error)
    parallel_builder.add_edge(Endpoint.start, "lookup")
    parallel_builder.add_edge(Endpoint.start, "value")

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(parallel_builder.compile().ainvoke({}))
    actual = {type(error) for error in raised.value.exceptions}
    assert actual == {LookupError, ValueError}, f"Expected the two node exceptions, got {actual}."


@pytest.mark.parametrize(
    ("route", "error"),
    [("missing", ValueError), (["first", "missing"], ValueError), (3, TypeError), (None, TypeError)],
    ids=["unknown-name", "sequence-with-unknown-name", "int", "none"],
)
def test_router_that_returns_an_invalid_route_raises(
    parallel_builder: StateGraph[State[int], State[int], State[int]], route: object, error: type[Exception]
) -> None:
    parallel_builder.add_conditional_edges("first", lambda state: route)

    with pytest.raises(error):
        asyncio.run(parallel_builder.compile().ainvoke({}))


@pytest.mark.parametrize(
    "add_invalid_edge",
    [
        lambda builder: builder.add_edge("first", "missing"),
        lambda builder: builder.add_edge("missing", "first"),
        lambda builder: builder.add_conditional_edges("missing", lambda state: Endpoint.end),
    ],
    ids=["unknown-target", "unknown-source", "unknown-router-source"],
)
def test_edge_of_an_unknown_node_is_rejected_by_compile(
    parallel_builder: StateGraph[State[int], State[int], State[int]],
    add_invalid_edge: Callable[[StateGraph[State[int], State[int], State[int]]], None],
) -> None:
    add_invalid_edge(parallel_builder)

    with pytest.raises(ValueError):
        parallel_builder.compile()


def test_graph_without_an_entry_edge_is_rejected_by_compile() -> None:
    builder = StateGraph(State[int])
    builder.add_node("only", lambda state: None)

    with pytest.raises(ValueError):
        builder.compile()


@pytest.mark.parametrize("name", ["first", "second"])
def test_duplicate_node_name_is_rejected(
    parallel_builder: StateGraph[State[int], State[int], State[int]], name: str
) -> None:
    with pytest.raises(ValueError):
        parallel_builder.add_node(name, lambda state: None)

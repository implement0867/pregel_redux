import asyncio
from collections.abc import Callable, Mapping
from operator import add
from typing import Annotated, Literal, TypedDict

import pytest

from pregel_redux import Command, Endpoint, Reducer, StateGraph


class State[T](TypedDict, total=False):
    items: Annotated[list[T], Reducer(add)]
    latest: T
    failures: int


class Input[T](TypedDict):
    items: Annotated[list[T], Reducer(add)]


class Output[T](TypedDict):
    items: Annotated[list[T], Reducer(add)]
    latest: T


class BareCallableState(TypedDict, total=False):
    items: Annotated[list[int], add]


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
    def route_work(state: State[str]) -> Command[Literal["work"]]:
        return Command(goto=Endpoint.end if state.get("items") else "work")

    def do_work(state: State[str]) -> State[str]:
        if results.pop(0):
            return {"items": ["done"]}
        return {"failures": state.get("failures", 0) + 1}

    builder = StateGraph(State[str])
    builder.add_node("router", route_work)
    builder.add_node("work", do_work)
    builder.add_edge(Endpoint.start, "router")
    builder.add_edge("work", "router")
    return builder


def add_edge_to_unknown_target(builder: StateGraph[State[int], State[int], State[int]]) -> None:
    builder.add_edge("first", "missing")


def add_edge_from_unknown_source(builder: StateGraph[State[int], State[int], State[int]]) -> None:
    builder.add_edge("missing", "first")


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


@pytest.mark.parametrize(
    ("schema", "update"),
    [(State[int], {"latest": 1}), (BareCallableState, {"items": [1]})],
    ids=["plain-field", "field-with-a-bare-callable"],
)
def test_parallel_writes_to_a_channel_without_reducer_are_rejected(
    schema: type[State[int] | BareCallableState], update: Mapping[str, object]
) -> None:
    """
    Arrange: two nodes of one step that write the same channel, which has no Reducer marker.
    Act: run the graph.
    Assert: a ValueError is raised.
    """
    builder = StateGraph(schema)
    builder.add_node("first", lambda state: update)
    builder.add_node("second", lambda state: update)
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
    "add_invalid_edge",
    [add_edge_to_unknown_target, add_edge_from_unknown_source],
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

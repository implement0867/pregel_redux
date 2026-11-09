import asyncio
from collections.abc import Callable
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


def build_sequential[T](produce: Callable[[], T]) -> StateGraph[State[T], Input[T], Output[T]]:
    def collect(state: State[T]) -> State[T]:
        return {"items": [produce()]}

    def remember(state: State[T]) -> State[T]:
        return {"latest": produce()}

    builder = StateGraph(State[T], input_schema=Input[T], output_schema=Output[T])
    builder.add_node("collect", collect)
    builder.add_node("remember", remember)
    builder.add_edge(Endpoint.start, "collect")
    builder.add_edge("collect", "remember")
    builder.add_edge("remember", Endpoint.end)
    return builder


def build_parallel[T](produce: Callable[[], T]) -> StateGraph[State[T], State[T], State[T]]:
    async def first(state: State[T]) -> State[T]:
        return {"items": [produce()]}

    async def second(state: State[T]) -> State[T]:
        return {"items": [produce()]}

    builder = StateGraph(State[T])
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start, "second")
    return builder


def test_sequential_partial_writes_to_a_generic_schema_keep_all_channels() -> None:
    graph = build_sequential(lambda: 1).compile()

    assert asyncio.run(graph.ainvoke({"items": [0]})) == {"items": [0, 1], "latest": 1}


def test_parallel_writes_to_a_generic_schema_use_the_reducer() -> None:
    graph = build_parallel(lambda: 1).compile()

    assert asyncio.run(graph.ainvoke({"items": [0], "latest": 0})) == {"items": [0, 1, 1], "latest": 0}


def test_parallel_writes_to_a_channel_without_reducer_are_rejected() -> None:
    def write_latest(state: State[int]) -> State[int]:
        return {"latest": 1}

    builder = StateGraph(State[int])
    builder.add_node("first", write_latest)
    builder.add_node("second", write_latest)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(Endpoint.start, "second")

    with pytest.raises(ValueError, match="more than one value"):
        asyncio.run(builder.compile().ainvoke({}))


def build_routed(results: list[bool]) -> StateGraph[State[str], State[str], State[str]]:
    def router(state: State[str]) -> None:
        return None

    def work(state: State[str]) -> State[str]:
        if results.pop(0):
            return {"items": ["done"]}
        return {"failures": state.get("failures", 0) + 1}

    def route(state: State[str]) -> str | Literal[Endpoint.end]:
        return Endpoint.end if state.get("items") else "work"

    builder = StateGraph(State[str])
    builder.add_node("router", router)
    builder.add_node("work", work)
    builder.add_edge(Endpoint.start, "router")
    builder.add_conditional_edges("router", route)
    builder.add_edge("work", "router")
    return builder


def test_router_runs_a_failed_node_again_as_a_routing_decision() -> None:
    graph = build_routed([False, False, True]).compile()

    assert asyncio.run(graph.ainvoke({})) == {"items": ["done"], "failures": 2}


def test_router_receives_the_writes_of_its_own_node() -> None:
    def write(state: State[int]) -> State[int]:
        return {"latest": 5}

    def route(state: State[int]) -> list[str | Literal[Endpoint.end]]:
        return ["record"] if state.get("latest") == 5 else [Endpoint.end]

    def record(state: State[int]) -> State[int]:
        return {"items": [state["latest"]]}

    builder = StateGraph(State[int])
    builder.add_node("write", write)
    builder.add_node("record", record)
    builder.add_edge(Endpoint.start, "write")
    builder.add_conditional_edges("write", route)

    assert asyncio.run(builder.compile().ainvoke({})) == {"latest": 5, "items": [5]}


def test_graph_that_does_not_finish_within_the_step_limit_raises() -> None:
    graph = build_routed([False] * 10).compile()

    with pytest.raises(RuntimeError, match="within 4 steps"):
        asyncio.run(graph.ainvoke({}, step_limit=4))


def test_exception_of_one_node_leaves_the_run_unchanged() -> None:
    failure = LookupError("no value")

    def fail(state: State[int]) -> State[int]:
        raise failure

    builder = build_parallel(lambda: 1)
    builder.add_node("fail", fail)
    builder.add_edge(Endpoint.start, "fail")

    with pytest.raises(LookupError) as raised:
        asyncio.run(builder.compile().ainvoke({}))
    assert raised.value is failure


def test_router_that_returns_an_unknown_node_raises() -> None:
    builder = build_parallel(lambda: 1)
    builder.add_conditional_edges("first", lambda state: "missing")

    with pytest.raises(ValueError, match="not nodes of this graph"):
        asyncio.run(builder.compile().ainvoke({}))


def test_edge_to_an_unknown_node_is_rejected_by_compile() -> None:
    builder = build_parallel(lambda: 1)
    builder.add_edge("first", "missing")

    with pytest.raises(ValueError, match="not nodes of this graph"):
        builder.compile()


def test_graph_without_an_entry_edge_is_rejected_by_compile() -> None:
    builder = StateGraph(State[int])
    builder.add_node("only", lambda state: None)

    with pytest.raises(ValueError, match=r"no edge from Endpoint\.start"):
        builder.compile()


def test_duplicate_node_name_is_rejected() -> None:
    builder = build_parallel(lambda: 1)

    with pytest.raises(ValueError, match="already in this graph"):
        builder.add_node("first", lambda state: None)

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Literal, TypedDict

import pytest

from pregel_redux import Depends, Endpoint, Scope, StateGraph


class State(TypedDict, total=False):
    seen: Annotated[list[int], add]


@dataclass
class Journal:
    events: list[str] = field(default_factory=list)
    opened: int = 0

    def open_resource(self, name: str) -> int:
        self.opened += 1
        self.events.append(f"open {name} {self.opened}")
        return self.opened


def build_two_node_graph(first: object, second: object, *, parallel: bool) -> StateGraph[State, State, State]:
    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(Endpoint.start, "first")
    builder.add_edge(*((Endpoint.start, "second") if parallel else ("first", "second")))
    return builder


def test_node_scoped_provider_opens_and_closes_for_each_node_call() -> None:
    journal = Journal()

    async def open_connection() -> AsyncIterator[int]:
        number = journal.open_resource("connection")
        yield number
        journal.events.append(f"close connection {number}")

    def use_value(state: State, number: Annotated[int, Depends(open_connection)]) -> State:
        journal.events.append(f"node {number}")
        return {"seen": [number]}

    graph = build_two_node_graph(use_value, use_value, parallel=False).compile()

    assert asyncio.run(graph.ainvoke({})) == {"seen": [1, 2]}
    assert journal.events == [
        "open connection 1",
        "node 1",
        "close connection 1",
        "open connection 2",
        "node 2",
        "close connection 2",
    ]


def test_run_scoped_provider_opens_one_time_and_closes_when_the_run_ends() -> None:
    journal = Journal()

    def open_client() -> Iterator[int]:
        number = journal.open_resource("client")
        yield number
        journal.events.append(f"close client {number}")

    async def use_value(state: State, number: Annotated[int, Depends(open_client, scope=Scope.run)]) -> State:
        journal.events.append("node")
        return {"seen": [number]}

    graph = build_two_node_graph(use_value, use_value, parallel=True).compile()

    assert asyncio.run(graph.ainvoke({})) == {"seen": [1, 1]}
    assert journal.events == ["open client 1", "node", "node", "close client 1"]
    assert asyncio.run(graph.ainvoke({})) == {"seen": [2, 2]}


def test_run_scoped_provider_of_a_node_that_never_runs_is_not_opened() -> None:
    journal = Journal()

    def open_client() -> int:
        return journal.open_resource("client")

    def use_value(state: State, number: Annotated[int, Depends(open_client, scope=Scope.run)]) -> State:
        return {"seen": [number]}

    builder = StateGraph(State)
    builder.add_node("start", lambda state: None)
    builder.add_node("unused", use_value)
    builder.add_edge(Endpoint.start, "start")

    assert asyncio.run(builder.compile().ainvoke({})) == {}
    assert journal.events == []


def test_provider_requested_two_times_in_one_node_call_runs_one_time() -> None:
    journal = Journal()

    async def provide_number() -> int:
        return journal.open_resource("number")

    def double_number(value: Annotated[int, Depends(provide_number)]) -> int:
        return value * 2

    def use_value(
        state: State, value: Annotated[int, Depends(provide_number)], twice: Annotated[int, Depends(double_number)]
    ) -> State:
        return {"seen": [value, twice]}

    builder = StateGraph(State)
    builder.add_node("use", use_value)
    builder.add_edge(Endpoint.start, "use")

    assert asyncio.run(builder.compile().ainvoke({})) == {"seen": [1, 2]}
    assert builder.dependencies == {provide_number, double_number}


def test_router_receives_dependencies_of_the_same_node_call() -> None:
    journal = Journal()

    def provide_number() -> int:
        return journal.open_resource("number")

    def use_value(state: State, value: Annotated[int, Depends(provide_number)]) -> State:
        return {"seen": [value]}

    def choose_route(state: State, value: Annotated[int, Depends(provide_number)]) -> Literal[Endpoint.end]:
        journal.events.append(f"route {value}")
        return Endpoint.end

    builder = StateGraph(State)
    builder.add_node("use", use_value)
    builder.add_edge(Endpoint.start, "use")
    builder.add_conditional_edges("use", choose_route)

    asyncio.run(builder.compile().ainvoke({}))
    assert journal.events == ["open number 1", "route 1"]


def test_provider_cleanup_receives_the_exception_of_the_node() -> None:
    caught: list[BaseException] = []

    def open_test_resource() -> Iterator[int]:
        try:
            yield 1
        except LookupError as error:
            caught.append(error)
            raise

    def raise_lookup_error(state: State, value: Annotated[int, Depends(open_test_resource)]) -> State:
        raise LookupError("no value")

    builder = StateGraph(State)
    builder.add_node("fail", raise_lookup_error)
    builder.add_edge(Endpoint.start, "fail")

    with pytest.raises(LookupError) as raised:
        asyncio.run(builder.compile().ainvoke({}))
    assert caught == [raised.value]


def test_override_replaces_a_provider_for_every_node() -> None:
    def provide_number() -> int:
        raise NotImplementedError

    def use_value(state: State, value: Annotated[int, Depends(provide_number)]) -> State:
        return {"seen": [value]}

    graph = build_two_node_graph(use_value, use_value, parallel=True).compile({provide_number: lambda: 7})

    assert asyncio.run(graph.ainvoke({})) == {"seen": [7, 7]}


def test_run_scoped_provider_that_requests_a_node_scoped_provider_is_rejected_by_compile() -> None:
    def provide_short_value() -> int:
        return 1

    def provide_long_value(value: Annotated[int, Depends(provide_short_value)]) -> int:
        return value

    def use_value(state: State, value: Annotated[int, Depends(provide_long_value, scope=Scope.run)]) -> None:
        return None

    builder = StateGraph(State)
    builder.add_node("use", use_value)
    builder.add_edge(Endpoint.start, "use")

    with pytest.raises(ValueError, match="requests node-scoped dependency"):
        builder.compile()


def test_parameter_without_depends_is_rejected_by_compile() -> None:
    def use_value(state: State, config: int) -> None:
        return None

    builder = StateGraph(State)
    builder.add_node("use", use_value)
    builder.add_edge(Endpoint.start, "use")

    with pytest.raises(TypeError, match=r"'config'.*not annotated with Depends"):
        builder.compile()


def test_dependency_cycle_is_rejected_by_compile() -> None:
    def provide_first(value: int) -> int:
        return value

    def provide_second(value: Annotated[int, Depends(provide_first)]) -> int:
        return value

    provide_first.__annotations__["value"] = Annotated[int, Depends(provide_second)]

    def use_value(state: State, value: Annotated[int, Depends(provide_first)]) -> None:
        return None

    builder = StateGraph(State)
    builder.add_node("use", use_value)
    builder.add_edge(Endpoint.start, "use")

    with pytest.raises(ValueError, match="dependency cycle"):
        builder.compile()


def test_one_provider_requested_with_two_scopes_gives_two_values() -> None:
    journal = Journal()

    def provide_number() -> int:
        return journal.open_resource("number")

    def use_run_value(state: State, value: Annotated[int, Depends(provide_number, scope=Scope.run)]) -> State:
        return {"seen": [value]}

    def use_node_value(state: State, value: Annotated[int, Depends(provide_number)]) -> State:
        return {"seen": [value]}

    graph = build_two_node_graph(use_run_value, use_node_value, parallel=False).compile()

    assert asyncio.run(graph.ainvoke({})) == {"seen": [1, 2]}

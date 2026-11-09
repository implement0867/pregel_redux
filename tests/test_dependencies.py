import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Literal, TypedDict

import pytest

from pregel_redux import Depends, Endpoint, Scope, StateGraph, provider


class State(TypedDict, total=False):
    seen: Annotated[list[int], add]


@dataclass
class Journal:
    events: list[str] = field(default_factory=list)
    opened: int = 0

    def open(self, name: str) -> int:
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

    async def connection() -> AsyncIterator[int]:
        number = journal.open("connection")
        yield number
        journal.events.append(f"close connection {number}")

    def use(state: State, number: Annotated[int, Depends(connection)]) -> State:
        journal.events.append(f"node {number}")
        return {"seen": [number]}

    graph = build_two_node_graph(use, use, parallel=False).compile()

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

    @provider(scope=Scope.run)
    def client() -> Iterator[int]:
        number = journal.open("client")
        yield number
        journal.events.append(f"close client {number}")

    async def use(state: State, number: Annotated[int, Depends(client)]) -> State:
        journal.events.append("node")
        return {"seen": [number]}

    graph = build_two_node_graph(use, use, parallel=True).compile()

    assert asyncio.run(graph.ainvoke({})) == {"seen": [1, 1]}
    assert journal.events == ["open client 1", "node", "node", "close client 1"]
    assert asyncio.run(graph.ainvoke({})) == {"seen": [2, 2]}


def test_run_scoped_provider_of_a_node_that_never_runs_is_not_opened() -> None:
    journal = Journal()

    @provider(scope=Scope.run)
    def client() -> int:
        return journal.open("client")

    def use(state: State, number: Annotated[int, Depends(client)]) -> State:
        return {"seen": [number]}

    builder = StateGraph(State)
    builder.add_node("start", lambda state: None)
    builder.add_node("unused", use)
    builder.add_edge(Endpoint.start, "start")

    assert asyncio.run(builder.compile().ainvoke({})) == {}
    assert journal.events == []


def test_provider_requested_two_times_in_one_node_call_runs_one_time() -> None:
    journal = Journal()

    async def number() -> int:
        return journal.open("number")

    def doubled(value: Annotated[int, Depends(number)]) -> int:
        return value * 2

    def use(state: State, value: Annotated[int, Depends(number)], twice: Annotated[int, Depends(doubled)]) -> State:
        return {"seen": [value, twice]}

    builder = StateGraph(State)
    builder.add_node("use", use)
    builder.add_edge(Endpoint.start, "use")

    assert asyncio.run(builder.compile().ainvoke({})) == {"seen": [1, 2]}
    assert builder.dependencies == {number, doubled}


def test_router_receives_dependencies_of_the_same_node_call() -> None:
    journal = Journal()

    def number() -> int:
        return journal.open("number")

    def use(state: State, value: Annotated[int, Depends(number)]) -> State:
        return {"seen": [value]}

    def route(state: State, value: Annotated[int, Depends(number)]) -> Literal[Endpoint.end]:
        journal.events.append(f"route {value}")
        return Endpoint.end

    builder = StateGraph(State)
    builder.add_node("use", use)
    builder.add_edge(Endpoint.start, "use")
    builder.add_conditional_edges("use", route)

    asyncio.run(builder.compile().ainvoke({}))
    assert journal.events == ["open number 1", "route 1"]


def test_provider_cleanup_receives_the_exception_of_the_node() -> None:
    caught: list[BaseException] = []

    def resource() -> Iterator[int]:
        try:
            yield 1
        except LookupError as error:
            caught.append(error)
            raise

    def fail(state: State, value: Annotated[int, Depends(resource)]) -> State:
        raise LookupError("no value")

    builder = StateGraph(State)
    builder.add_node("fail", fail)
    builder.add_edge(Endpoint.start, "fail")

    with pytest.raises(LookupError) as raised:
        asyncio.run(builder.compile().ainvoke({}))
    assert caught == [raised.value]


def test_override_replaces_a_provider_for_every_node() -> None:
    def number() -> int:
        raise NotImplementedError

    def use(state: State, value: Annotated[int, Depends(number)]) -> State:
        return {"seen": [value]}

    graph = build_two_node_graph(use, use, parallel=True).compile({number: lambda: 7})

    assert asyncio.run(graph.ainvoke({})) == {"seen": [7, 7]}


def test_run_scoped_provider_that_requests_a_node_scoped_provider_is_rejected_by_compile() -> None:
    def short() -> int:
        return 1

    @provider(scope=Scope.run)
    def long(value: Annotated[int, Depends(short)]) -> int:
        return value

    def use(state: State, value: Annotated[int, Depends(long)]) -> None:
        return None

    builder = StateGraph(State)
    builder.add_node("use", use)
    builder.add_edge(Endpoint.start, "use")

    with pytest.raises(ValueError, match="requests node-scoped provider"):
        builder.compile()


def test_parameter_without_depends_is_rejected_by_compile() -> None:
    def use(state: State, config: int) -> None:
        return None

    builder = StateGraph(State)
    builder.add_node("use", use)
    builder.add_edge(Endpoint.start, "use")

    with pytest.raises(TypeError, match=r"'config'.*not annotated with Depends"):
        builder.compile()


def test_dependency_cycle_is_rejected_by_compile() -> None:
    def first(value: int) -> int:
        return value

    def second(value: Annotated[int, Depends(first)]) -> int:
        return value

    first.__annotations__["value"] = Annotated[int, Depends(second)]

    def use(state: State, value: Annotated[int, Depends(first)]) -> None:
        return None

    builder = StateGraph(State)
    builder.add_node("use", use)
    builder.add_edge(Endpoint.start, "use")

    with pytest.raises(ValueError, match="dependency cycle"):
        builder.compile()

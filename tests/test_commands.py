import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

import pytest
from pydantic import BaseModel

from pregel_redux import Command, Depends, Endpoint, Reducer, Send, StateDep, StateGraph
from pregel_redux.commands import resolve_goto
from tests.conftest import Journal, join_lists

type Goto = str | Literal[Endpoint.end] | Send | list[str | Literal[Endpoint.end] | Send]


class State(TypedDict, total=False):
    topics: list[str]
    jokes: Annotated[list[str], Reducer(join_lists, initial_factory=list)]
    visits: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class TopicsInput(TypedDict, total=False):
    topics: list[str]


class DictRequest(TypedDict):
    topic: str


@dataclass
class DataclassRequest:
    topic: str


class ModelRequest(BaseModel):
    topic: str


def visit_work(state: State) -> State:
    return {"visits": ["work"]}


def visit_review(state: State) -> State:
    return {"visits": ["review"]}


def write_dict_joke(request: DictRequest) -> State:
    return {"jokes": [f"a joke about {request['topic']}"]}


def write_dataclass_joke(request: DataclassRequest) -> State:
    return {"jokes": [f"a joke about {request.topic}"]}


def write_model_joke(request: ModelRequest) -> State:
    return {"jokes": [f"a joke about {request.topic}"]}


def plan_jokes(state: State) -> Command[str]:
    return Command(goto=[Send("write_joke", {"topic": topic}) for topic in state.get("topics", [])])


def declare_missing_target(state: State) -> Command[Literal["work", "missing"]]:
    return Command(goto="work")


def declare_missing_target_in_a_union(state: State) -> Command[Literal["missing"]] | State:
    return {}


def declare_missing_target_in_two_literals(state: State) -> Command[Literal["work", "missing"]]:
    return Command(goto="work")


def build_planned_graph(plan: Callable[[State], object]) -> StateGraph[State, State, State]:
    builder = StateGraph(State)
    builder.add_node("plan", plan)
    builder.add_node("work", visit_work)
    builder.add_node("review", visit_review)
    builder.add_edge(Endpoint.start, "plan")
    return builder


def build_joke_graph(write_joke: Callable[..., object], input_schema: type[object]) -> StateGraph[State, State, State]:
    builder = StateGraph(State)
    builder.add_node("plan", plan_jokes)
    builder.add_node("write_joke", write_joke, input_schema=input_schema)
    builder.add_edge(Endpoint.start, "plan")
    return builder


@pytest.mark.parametrize(
    ("goto", "expected"),
    [
        ("work", ["work"]),
        (["work", "review"], ["review", "work"]),
        (["work", "work"], ["work"]),
        (Endpoint.end, []),
        ([], []),
        (["work", Endpoint.end], ["work"]),
    ],
    ids=["one-name", "two-names", "one-name-two-times", "end", "empty", "name-and-end"],
)
def test_goto_starts_the_named_nodes(goto: Goto, expected: list[str]) -> None:
    graph = build_planned_graph(lambda state: Command[str](goto=goto)).compile()

    actual = asyncio.run(graph.ainvoke({}))

    assert actual.get("visits", []) == expected, f"Expected the visits {expected}, got {actual}."


@pytest.mark.parametrize(
    "update",
    [{"visits": ["plan"]}, {"visits": ["plan"], "topics": ["cats"]}, {}, None],
    ids=["one-channel", "two-channels", "empty", "none"],
)
def test_command_with_an_update_equals_the_same_update(update: State | None) -> None:
    expected = asyncio.run(build_planned_graph(lambda state: update).compile().ainvoke({"visits": ["input"]}))

    actual = asyncio.run(
        build_planned_graph(lambda state: Command[str](update=update)).compile().ainvoke({"visits": ["input"]})
    )

    assert actual == expected, f"Expected {expected}, got {actual}. A command must apply its update as a plain update."


@pytest.mark.parametrize(
    ("goto", "expected"),
    [("review", ["review", "work"]), ("work", ["work"]), (Endpoint.end, ["work"])],
    ids=["other-node", "same-node", "end"],
)
def test_goto_adds_to_the_static_edges_of_the_node(goto: Goto, expected: list[str]) -> None:
    builder = build_planned_graph(lambda state: Command[str](update={"visits": []}, goto=goto))
    builder.add_edge("plan", "work")

    actual = asyncio.run(builder.compile().ainvoke({}))

    assert actual.get("visits", []) == expected, f"Expected the visits {expected}, got {actual}."


@pytest.mark.parametrize(
    "goto",
    ["missing", ["work", "missing"], Send("missing", {}), ["work", Send("missing", {})]],
    ids=["name", "name-in-a-list", "send", "send-in-a-list"],
)
def test_goto_to_an_unknown_node_raises(goto: Goto) -> None:
    graph = build_planned_graph(lambda state: Command[str](goto=goto)).compile()

    with pytest.raises(ValueError):
        asyncio.run(graph.ainvoke({}))


@pytest.mark.parametrize("goto", [3, None, 1.5, [3], Endpoint.start, ["work", Endpoint.start]], ids=repr)
def test_goto_of_an_unsupported_type_is_rejected(goto: object) -> None:
    with pytest.raises(TypeError):
        resolve_goto(goto, ["work"])


@pytest.mark.parametrize(
    "plan", [declare_missing_target, declare_missing_target_in_a_union, declare_missing_target_in_two_literals]
)
def test_declared_target_that_is_not_a_node_is_rejected_by_compile(plan: Callable[[State], object]) -> None:
    builder = build_planned_graph(plan)

    with pytest.raises(ValueError):
        builder.compile()


@pytest.mark.parametrize(
    ("topics", "expected"),
    [
        ([], []),
        (["cats"], ["a joke about cats"]),
        (["cats", "dogs", "birds"], ["a joke about cats", "a joke about dogs", "a joke about birds"]),
        (["cats", "cats"], ["a joke about cats", "a joke about cats"]),
    ],
    ids=["no-topic", "one-topic", "three-topics", "one-topic-two-times"],
)
@pytest.mark.parametrize(
    ("write_joke", "input_schema"),
    [(write_dict_joke, DictRequest), (write_dataclass_joke, DataclassRequest), (write_model_joke, ModelRequest)],
    ids=["typed-dict", "dataclass", "model"],
)
def test_each_send_starts_the_node_with_its_payload(
    write_joke: Callable[..., object], input_schema: type[object], topics: list[str], expected: list[str]
) -> None:
    graph = build_joke_graph(write_joke, input_schema).compile()

    actual = asyncio.run(graph.ainvoke({"topics": topics}))

    assert actual.get("jokes", []) == expected, f"Expected the jokes {expected}, got {actual}."


@pytest.mark.parametrize(
    ("input_schema", "payload", "expected"),
    [
        (DictRequest, {"topic": "cats"}, {"topic": "cats"}),
        (DataclassRequest, {"topic": "cats"}, DataclassRequest(topic="cats")),
        (ModelRequest, {"topic": "cats"}, ModelRequest(topic="cats")),
        (DataclassRequest, DataclassRequest(topic="cats"), DataclassRequest(topic="cats")),
        (ModelRequest, ModelRequest(topic="cats"), ModelRequest(topic="cats")),
    ],
    ids=["typed-dict", "dataclass", "model", "dataclass-instance", "model-instance"],
)
def test_send_gives_the_payload_as_the_input_type_of_the_node(
    input_schema: type[object], payload: object, expected: object
) -> None:
    received: list[object] = []
    builder = StateGraph(State)
    builder.add_node("plan", lambda state: Command[str](goto=Send("receive", payload)))
    builder.add_node("receive", received.append, input_schema=input_schema)
    builder.add_edge(Endpoint.start, "plan")

    asyncio.run(builder.compile().ainvoke({}))

    assert received == [expected], f"Expected the node to receive {expected!r}, got {received}."


@pytest.mark.parametrize("payload", [{"topic": ["cats"]}, {}, {"subject": "cats"}], ids=repr)
def test_model_input_type_checks_the_payload_of_a_send(payload: Mapping[str, object]) -> None:
    builder = StateGraph(State)
    builder.add_node("plan", lambda state: Command[str](goto=Send("write_joke", payload)))
    builder.add_node("write_joke", write_model_joke, input_schema=ModelRequest)
    builder.add_edge(Endpoint.start, "plan")

    with pytest.raises(ValueError):
        asyncio.run(builder.compile().ainvoke({}))


@pytest.mark.parametrize(
    "payload",
    [{"topic": ["cats"]}, {"topic": None}, {"subject": "cats"}, {"topic": "cats", "subject": "cats"}, "cats"],
    ids=["wrong-type", "none", "unknown-field", "known-and-unknown-field", "not-a-mapping"],
)
@pytest.mark.parametrize(
    ("write_joke", "input_schema"),
    [(write_dict_joke, DictRequest), (write_dataclass_joke, DataclassRequest), (write_model_joke, ModelRequest)],
    ids=["typed-dict", "dataclass", "model"],
)
def test_send_with_a_payload_that_its_channels_cannot_hold_is_rejected(
    write_joke: Callable[..., object], input_schema: type[object], payload: object
) -> None:
    """
    Arrange: a node that sends a payload with a value of the wrong type, an unknown field, or no fields.
    Act: run the graph.
    Assert: a ValueError or a TypeError is raised, and the target node does not run.
    """
    received: list[object] = []

    def record_request(request: object) -> None:
        received.append(request)

    builder = StateGraph(State)
    builder.add_node("plan", lambda state: Command[str](goto=Send("write_joke", payload)))
    builder.add_node("write_joke", record_request, input_schema=input_schema)
    builder.add_edge(Endpoint.start, "plan")

    with pytest.raises((ValueError, TypeError)):
        asyncio.run(builder.compile().ainvoke({}))

    assert received == [], f"Expected the target node not to run, got the requests {received}."


@pytest.mark.parametrize("topics", [["cats"], ["cats", "dogs"], []], ids=repr)
def test_send_to_a_node_without_an_input_type_gives_the_payload_as_the_state(topics: list[str]) -> None:
    def copy_topics(state: State) -> State:
        return {"visits": state.get("topics", [])}

    builder = StateGraph(State)
    builder.add_node("plan", lambda state: Command[str](goto=Send("copy", {"topics": topics})))
    builder.add_node("copy", copy_topics)
    builder.add_edge(Endpoint.start, "plan")

    actual = asyncio.run(builder.compile().ainvoke({"topics": ["input"]}))

    assert actual.get("visits", []) == topics, f"Expected the node to read {topics} from the payload, got {actual}."


@pytest.mark.parametrize(
    ("input_state", "expected"),
    [
        ({"topics": ["cats"], "visits": ["input"]}, {"topics": ["cats"]}),
        ({"visits": ["input"]}, {}),
        ({}, {}),
    ],
    ids=["field-with-a-value", "field-without-a-value", "empty-state"],
)
def test_edge_gives_a_node_the_fields_of_its_input_type(input_state: State, expected: TopicsInput) -> None:
    received: list[TopicsInput] = []

    def receive_topics(request: TopicsInput) -> None:
        received.append(request)

    builder = StateGraph(State)
    builder.add_node("receive", receive_topics, input_schema=TopicsInput)
    builder.add_edge(Endpoint.start, "receive")

    asyncio.run(builder.compile().ainvoke(input_state))

    assert received == [expected], f"Expected the node to receive {expected}, got {received}."


@pytest.mark.parametrize("count", [1, 2, 3])
def test_each_send_opens_its_own_node_scoped_dependencies(journal: Journal, count: int) -> None:
    """
    Arrange: a node with a node-scoped provider that receives several sends in one step.
    Act: run the graph.
    Assert: each send received its own value of the provider.
    """

    def provide_number() -> int:
        return journal.open_resource("number")

    def write_number(request: DictRequest, number: Annotated[int, Depends(provide_number)]) -> State:
        return {"jokes": [f"{request['topic']} {number}"]}

    builder = StateGraph(State)
    builder.add_node("plan", lambda state: Command[str](goto=[Send("write", {"topic": "cats"})] * count))
    builder.add_node("write", write_number, input_schema=DictRequest)
    builder.add_edge(Endpoint.start, "plan")
    expected = [f"cats {index}" for index in range(1, count + 1)]

    actual = asyncio.run(builder.compile().ainvoke({}))

    assert sorted(actual.get("jokes", [])) == expected, f"Expected one dependency value for each send, got {actual}."


@pytest.mark.parametrize("topic", ["cats", "dogs"])
def test_provider_receives_the_payload_of_a_send_as_the_state(topic: str) -> None:
    def read_topic(request: StateDep[DictRequest]) -> str:
        return request["topic"]

    def write_topic(request: DictRequest, name: Annotated[str, Depends(read_topic)]) -> State:
        return {"jokes": [name]}

    builder = StateGraph(State)
    builder.add_node("plan", lambda state: Command[str](goto=Send("write", {"topic": topic})))
    builder.add_node("write", write_topic, input_schema=DictRequest)
    builder.add_edge(Endpoint.start, "plan")

    actual = asyncio.run(builder.compile().ainvoke({}))

    assert actual.get("jokes", []) == [topic], f"Expected the provider to read {topic!r}, got {actual}."

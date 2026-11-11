import asyncio
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Annotated, Literal, TypedDict

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from typing_extensions import TypeForm

from pregel_redux import (
    Answer,
    Checkpointer,
    CheckpointSource,
    Command,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupted,
    Reducer,
    RunResult,
    Send,
    StateGraph,
    Thread,
    interrupt,
)
from pregel_redux.checkpoints import TaskAddress, TaskState
from tests.conftest import Journal, append_item, join_lists


class State(TypedDict, total=False):
    items: Annotated[list[str], Reducer(join_lists, initial_factory=list)]
    topic: str


class Topic(BaseModel):
    name: str


class DictRequest(TypedDict):
    topic: str


@dataclass
class DataclassRequest:
    topic: str
    tags: tuple[str, ...] = ()


class ModelRequest(BaseModel):
    topic: str


class AliasRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    topic: str = Field(alias="subject")


@dataclass
class HiddenRequest:
    topic: str
    count: int = field(init=False, default=0)


class Animal(BaseModel):
    name: str


class Dog(Animal):
    sound: str


class PetState(TypedDict, total=False):
    pet: Animal


type Builder = StateGraph[State, State, State]


def run_until_done[OutputT](result: RunResult[OutputT]) -> OutputT:
    match result:
        case Finished(output=output):
            return output
        case Interrupted(interrupts=interrupts):
            raise AssertionError(f"Expected the run to finish, it stopped at {interrupts!r}.")


def read_interrupted[OutputT](result: RunResult[OutputT]) -> Interrupted:
    match result:
        case Interrupted():
            return result
        case Finished(output=output):
            raise AssertionError(f"Expected the run to stop at an interrupt, it finished with {output!r}.")


def build_writer(journal: Journal, name: str) -> Callable[[State], State]:
    def write(state: State) -> State:
        journal.events.append(name)
        return {"items": [name]}

    return write


def build_flaky_writer(journal: Journal, name: str, failures: list[Exception]) -> Callable[[State], State]:
    def write(state: State) -> State:
        journal.events.append(name)
        if failures:
            raise failures.pop(0)
        return {"items": [name]}

    return write


def build_asker(journal: Journal, name: str, questions: int) -> Callable[[State], State]:
    def ask(state: State) -> State:
        journal.events.append(name)
        answers = [interrupt({"node": name, "question": index}, str) for index in range(questions)]
        return {"items": [f"{name} {' '.join(answers)}"]}

    return ask


def build_chain(journal: Journal, names: list[str]) -> Builder:
    builder = StateGraph(State)
    for name in names:
        builder.add_node(name, build_writer(journal, name))
    builder.add_edge(Endpoint.start, names[0])
    for source, target in pairwise(names):
        builder.add_edge(source, target)
    return builder


def wrap_in_children(child: Builder, depth: int) -> Builder:
    graph = child
    for _ in range(depth):
        parent = StateGraph(State)
        parent.add_node("child", graph)
        parent.add_edge(Endpoint.start, "child")
        graph = parent
    return graph


def build_thread(builder: Builder, checkpointer: Checkpointer | None = None) -> Thread[State, State, State]:
    return Thread(
        graph=builder.compile(),
        checkpointer=InMemoryCheckpointer() if checkpointer is None else checkpointer,
        thread_id="thread",
    )


@pytest.mark.parametrize("names", [["first"], ["first", "second"], ["first", "second", "third"]], ids=len)
def test_thread_saves_one_checkpoint_for_the_input_and_one_after_each_step(journal: Journal, names: list[str]) -> None:
    thread = build_thread(build_chain(journal, names))
    expected = [*([CheckpointSource.loop] * len(names)), CheckpointSource.input]

    output = run_until_done(thread.invoke({"items": ["input"]}))
    history = thread.history()

    actual = [snapshot.checkpoint.source for snapshot in history]
    assert actual == expected, f"Expected the sources {expected}, got {actual}."
    parents = [snapshot.checkpoint.parent_id for snapshot in history]
    assert parents == [*(snapshot.checkpoint.id for snapshot in history[1:]), None], "Each checkpoint must follow one."
    assert history[0].values == output == {"items": ["input", *names]}, f"Expected the output state, got {output}."


@pytest.mark.parametrize("inputs", [[["a"], ["b"]], [["a"], [], ["c"]]], ids=len)
def test_each_run_of_a_thread_starts_from_the_state_of_the_last_run(journal: Journal, inputs: list[list[str]]) -> None:
    thread = build_thread(build_chain(journal, ["node"]))
    expected = [item for items in inputs for item in [*items, "node"]]

    outputs = [run_until_done(thread.invoke({"items": items})) for items in inputs]

    assert outputs[-1] == {"items": expected}, f"Expected the items {expected}, got {outputs[-1]}."
    steps = [snapshot.checkpoint.step for snapshot in thread.history()]
    assert steps == list(range(len(steps)))[::-1], f"Expected one chain of steps, got {steps}."


@pytest.mark.parametrize("depth", [0, 1, 2], ids=["graph", "child", "grandchild"])
@pytest.mark.parametrize("failure", [LookupError("no value"), ValueError("wrong value")], ids=repr)
def test_resume_after_a_failure_runs_only_the_tasks_that_did_not_finish(
    journal: Journal, depth: int, failure: Exception
) -> None:
    inner = StateGraph(State)
    inner.add_node("first", build_writer(journal, "first"))
    inner.add_node("steady", build_writer(journal, "steady"))
    inner.add_node("flaky", build_flaky_writer(journal, "flaky", [failure]))
    inner.add_edge(Endpoint.start, "first")
    inner.add_edge("first", "steady")
    inner.add_edge("first", "flaky")
    builder = StateGraph(State)
    builder.add_node("before", build_writer(journal, "before"))
    builder.add_node("inner", wrap_in_children(inner, depth))
    builder.add_edge(Endpoint.start, "before")
    builder.add_edge("before", "inner")
    thread = build_thread(builder)

    with pytest.raises(type(failure)):
        thread.invoke({})
    output = run_until_done(thread.resume())

    assert sorted(journal.events) == ["before", "first", "flaky", "flaky", "steady"], (
        f"Expected only the failed node to run again, got {journal.events}."
    )
    assert sorted(output.get("items", [])).count("steady") == 1, f"Expected one write of each node, got {output}."


class TopicState(TypedDict, total=False):
    topics: Annotated[list[Topic], Reducer(append_item, initial_factory=list)]


def build_topic_writer(name: str) -> Callable[[TopicState], dict[str, Topic]]:
    return lambda state: {"topics": Topic(name=name)}


@pytest.mark.parametrize("names", [["a"], ["a", "b"]], ids=len)
def test_saved_write_reaches_the_reducer_as_its_update_type_after_a_resume(names: list[str]) -> None:
    failures: list[Exception] = [LookupError("fail one time")]

    def fail_once(state: TopicState) -> TopicState:
        if failures:
            raise failures.pop(0)
        return {}

    builder = StateGraph(TopicState)
    for name in names:
        builder.add_node(name, build_topic_writer(name))
        builder.add_edge(Endpoint.start, name)
    builder.add_node("flaky", fail_once)
    builder.add_edge(Endpoint.start, "flaky")
    thread = Thread(graph=builder.compile(), checkpointer=InMemoryCheckpointer(), thread_id="thread")
    with pytest.raises(LookupError):
        thread.invoke({})

    output = thread.resume()

    expected = Finished(TopicState(topics=[Topic(name=name) for name in names]))
    assert output == expected, f"Expected the saved writes as topics, got {output}."


def test_resume_of_a_thread_that_finished_gives_its_output(journal: Journal) -> None:
    thread = build_thread(build_chain(journal, ["first", "second"]))
    expected = run_until_done(thread.invoke({}))

    actual = run_until_done(thread.resume())

    assert actual == expected, f"Expected the output {expected}, got {actual}."
    assert journal.events == ["first", "second"], f"Expected no node to run again, got {journal.events}."


@pytest.mark.parametrize(
    ("answer_type", "value", "expected"),
    [
        (bool, True, True),
        (int, 3, 3),
        (Literal["yes", "no"], "no", "no"),
        (tuple[int, int], [1, 2], (1, 2)),
        (Topic, {"name": "cats"}, Topic(name="cats")),
    ],
    ids=["bool", "int", "literal", "tuple", "model"],
)
def test_interrupt_gives_the_answer_of_the_resume_as_its_answer_type(
    answer_type: TypeForm[object], value: JsonValue, expected: object
) -> None:
    received: list[object] = []

    def ask(state: State) -> State:
        received.append(interrupt({"question": "what?"}, answer_type))
        return {}

    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "ask")
    thread = build_thread(builder)

    interrupted = read_interrupted(thread.invoke({}))
    run_until_done(thread.resume(Answer(interrupt_id=interrupted.interrupts[0].interrupt_id, value=value)))

    assert received == [expected], f"Expected the node to receive {expected!r}, got {received}."


def test_interrupt_in_an_async_node_resumes_with_the_answer() -> None:
    async def ask(state: State) -> State:
        await asyncio.sleep(0)
        return {"items": [interrupt("topic?", str)]}

    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_edge(Endpoint.start, "ask")
    thread = build_thread(builder)

    async def ask_and_answer() -> RunResult[State]:
        interrupted = read_interrupted(await thread.ainvoke({}))
        return await thread.aresume(Answer(interrupt_id=interrupted.interrupts[0].interrupt_id, value="cats"))

    output = run_until_done(asyncio.run(ask_and_answer()))

    assert output == {"items": ["cats"]}, f"Expected the answer in the output, got {output}."


def test_saved_interrupt_can_be_answered_by_another_thread_object(journal: Journal) -> None:
    builder = StateGraph(State)
    builder.add_node("ask", build_asker(journal, "ask", 1))
    builder.add_edge(Endpoint.start, "ask")
    checkpointer = InMemoryCheckpointer()
    interrupted = read_interrupted(build_thread(builder, checkpointer).invoke({}))
    second = build_thread(builder, InMemoryCheckpointer(deepcopy(checkpointer.threads)))

    state = second.get_state()
    output = run_until_done(second.resume(Answer(interrupt_id=interrupted.interrupts[0].interrupt_id, value="cats")))

    assert state is not None, "Expected the saved state."
    assert state.interrupts == interrupted.interrupts, f"Expected the saved interrupts, got {state.interrupts}."
    assert state.next == ("ask",), f"Expected the interrupted node to be next, got {state.next}."
    assert output == {"items": ["ask cats"]}, f"Expected the answer in the output, got {output}."


@pytest.mark.parametrize("questions", [2, 3])
def test_node_with_several_interrupts_receives_all_answers_in_order(journal: Journal, questions: int) -> None:
    builder = StateGraph(State)
    builder.add_node("ask", build_asker(journal, "ask", questions))
    builder.add_edge(Endpoint.start, "ask")
    thread = build_thread(builder)
    result = thread.invoke({})
    indexes: list[int] = []

    for answer in range(questions):
        pending = read_interrupted(result).interrupts[0]
        indexes.append(pending.interrupt_id.index)
        result = thread.resume(Answer(interrupt_id=pending.interrupt_id, value=str(answer)))

    expected = f"ask {' '.join(str(answer) for answer in range(questions))}"
    assert run_until_done(result) == {"items": [expected]}, f"Expected {expected!r}, got {result}."
    assert indexes == list(range(questions)), f"Expected one interrupt index after another, got {indexes}."


def test_interrupt_without_an_answer_keeps_its_node_waiting(journal: Journal) -> None:
    builder = StateGraph(State)
    builder.add_node("left", build_asker(journal, "left", 1))
    builder.add_node("right", build_asker(journal, "right", 1))
    builder.add_node("steady", build_writer(journal, "steady"))
    builder.add_edge(Endpoint.start, "left")
    builder.add_edge(Endpoint.start, "right")
    builder.add_edge(Endpoint.start, "steady")
    thread = build_thread(builder)
    left, right = read_interrupted(thread.invoke({})).interrupts

    waiting = read_interrupted(thread.resume(Answer(interrupt_id=left.interrupt_id, value="a"))).interrupts
    output = run_until_done(thread.resume(Answer(interrupt_id=right.interrupt_id, value="b")))

    assert waiting == (right,), f"Expected only the interrupt without an answer, got {waiting}."
    assert sorted(journal.events) == ["left", "left", "right", "right", "steady"], (
        f"Expected each asking node to run one more time, got {journal.events}."
    )
    assert sorted(output.get("items", [])) == ["left a", "right b", "steady"], f"Expected all writes, got {output}."


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_interrupt_inside_a_child_graph_resumes_inside_the_child_graph(journal: Journal, depth: int) -> None:
    inner = StateGraph(State)
    inner.add_node("before", build_writer(journal, "before"))
    inner.add_node("ask", build_asker(journal, "ask", 1))
    inner.add_edge(Endpoint.start, "before")
    inner.add_edge("before", "ask")
    thread = build_thread(wrap_in_children(inner, depth))
    interrupted = read_interrupted(thread.invoke({}))

    output = run_until_done(thread.resume(Answer(interrupt_id=interrupted.interrupts[0].interrupt_id, value="yes")))

    assert journal.events == ["before", "ask", "ask"], (
        f"Expected the child to resume at its node, got {journal.events}."
    )
    assert output.get("items", [])[-1] == "ask yes", f"Expected the answer in the output, got {output}."


def build_single_asker(journal: Journal) -> Thread[State, State, State]:
    builder = StateGraph(State)
    builder.add_node("ask", build_asker(journal, "ask", 1))
    builder.add_edge(Endpoint.start, "ask")
    return build_thread(builder)


def answer_twice(thread: Thread[State, State, State], interrupted: Interrupted) -> RunResult[State]:
    answer = Answer(interrupt_id=interrupted.interrupts[0].interrupt_id, value="a")
    return thread.resume(answer, answer)


def answer_an_unknown_interrupt(thread: Thread[State, State, State], interrupted: Interrupted) -> RunResult[State]:
    interrupt_id = interrupted.interrupts[0].interrupt_id
    return thread.resume(Answer(interrupt_id=interrupt_id.model_copy(update={"index": 1}), value="a"))


def answer_an_old_interrupt(thread: Thread[State, State, State], interrupted: Interrupted) -> RunResult[State]:
    answer = Answer(interrupt_id=interrupted.interrupts[0].interrupt_id, value="a")
    thread.resume(answer)
    return thread.resume(answer)


@pytest.mark.parametrize("answer", [answer_twice, answer_an_unknown_interrupt, answer_an_old_interrupt])
def test_answer_that_does_not_match_one_pending_interrupt_is_rejected(
    journal: Journal, answer: Callable[[Thread[State, State, State], Interrupted], RunResult[State]]
) -> None:
    thread = build_single_asker(journal)
    interrupted = read_interrupted(thread.invoke({}))

    with pytest.raises(ValueError):
        answer(thread, interrupted)


def test_resume_of_a_thread_without_a_checkpoint_is_rejected() -> None:
    thread = build_thread(build_chain(Journal(), ["node"]))

    with pytest.raises(LookupError):
        thread.resume()


def test_answer_of_the_wrong_type_leaves_the_interrupt_pending(journal: Journal) -> None:
    def ask_number(state: State) -> State:
        return {"items": [str(interrupt("how many?", int))]}

    builder = StateGraph(State)
    builder.add_node("ask", ask_number)
    builder.add_edge(Endpoint.start, "ask")
    thread = build_thread(builder)
    pending = read_interrupted(thread.invoke({})).interrupts[0]

    with pytest.raises(ValidationError):
        thread.resume(Answer(interrupt_id=pending.interrupt_id, value="many"))
    state = thread.get_state()
    output = run_until_done(thread.resume(Answer(interrupt_id=pending.interrupt_id, value=2)))

    assert state is not None, "Expected the saved state."
    assert state.interrupts == (pending,), f"Expected the interrupt to stay pending, got {state.interrupts}."
    assert output == {"items": ["2"]}, f"Expected the second answer in the output, got {output}."


def test_run_without_a_thread_cannot_stop_at_an_interrupt(journal: Journal) -> None:
    builder = StateGraph(State)
    builder.add_node("ask", build_asker(journal, "ask", 1))
    builder.add_edge(Endpoint.start, "ask")

    with pytest.raises(RuntimeError):
        builder.compile().invoke({})


def test_interrupt_outside_of_a_node_is_rejected() -> None:
    with pytest.raises(RuntimeError):
        interrupt("question", str)


@pytest.mark.parametrize(
    ("update", "as_node", "expected_next", "expected_items"),
    [
        ({"items": ["edit"]}, None, ("ask",), ["edit"]),
        ({"items": ["edit"]}, "ask", ("after",), ["edit"]),
        (Command[str](update={"items": ["edit"]}, goto="other"), "ask", ("after", "other"), ["edit"]),
        (Command[str](goto=Send("other", {"topic": "cats"})), "ask", ("after", "other"), []),
    ],
    ids=["keep-the-tasks", "as-a-node", "command-as-a-node", "send-as-a-node"],
)
def test_update_state_writes_the_update_and_plans_the_next_tasks(
    journal: Journal, update: object, as_node: str | None, expected_next: tuple[str, ...], expected_items: list[str]
) -> None:
    builder = StateGraph(State)
    builder.add_node("ask", build_asker(journal, "ask", 1))
    builder.add_node("after", wrap_in_children(build_chain(journal, ["after"]), 1))
    builder.add_node("other", build_writer(journal, "other"))
    builder.add_edge(Endpoint.start, "ask")
    builder.add_edge("ask", "after")
    thread = build_thread(builder)
    read_interrupted(thread.invoke({}))

    checkpoint_id = thread.update_state(update, as_node=as_node)
    state = thread.get_state()

    assert state is not None, "Expected the saved state."
    assert state.checkpoint.id == checkpoint_id, "Expected the update to be the last checkpoint."
    assert state.checkpoint.source is CheckpointSource.update, f"Expected an update, got {state.checkpoint.source}."
    assert state.next == expected_next, f"Expected the next nodes {expected_next}, got {state.next}."
    assert state.values == {"items": expected_items}, f"Expected the items {expected_items}, got {state.values}."


def test_resume_after_an_update_runs_the_planned_tasks_with_the_updated_values(journal: Journal) -> None:
    received: list[str] = []

    def read_topic(state: State) -> State:
        received.append(state.get("topic", ""))
        return {}

    builder = StateGraph(State)
    builder.add_node("fail", build_flaky_writer(journal, "fail", [LookupError("no topic")]))
    builder.add_node("read", read_topic)
    builder.add_edge(Endpoint.start, "fail")
    builder.add_edge("fail", "read")
    thread = build_thread(builder)
    with pytest.raises(LookupError):
        thread.invoke({"topic": "old"})

    thread.update_state({"topic": "new"})
    run_until_done(thread.resume())

    assert received == ["new"], f"Expected the node to read the updated topic, got {received}."


@pytest.mark.parametrize("updated_step", [0, 1])
def test_update_of_an_earlier_checkpoint_starts_a_branch_that_resume_continues(
    journal: Journal, updated_step: int
) -> None:
    names = ["first", "second"]
    thread = build_thread(build_chain(journal, names))
    run_until_done(thread.invoke({}))
    updated = next(snapshot for snapshot in thread.history() if snapshot.checkpoint.step == updated_step)
    journal.events.clear()

    thread.update_state({"items": ["edit"]}, checkpoint_id=updated.checkpoint.id)
    output = run_until_done(thread.resume())

    expected = [*names[:updated_step], "edit", *names[updated_step:]]
    assert output == {"items": expected}, f"Expected the branch to continue from the update, got {output}."
    assert journal.events == names[updated_step:], f"Expected the later nodes to run again, got {journal.events}."


def update_as_an_unknown_node(thread: Thread[State, State, State]) -> object:
    return thread.update_state({"items": ["edit"]}, as_node="missing")


def update_an_unknown_channel(thread: Thread[State, State, State]) -> object:
    return thread.update_state({"missing": 1})


@pytest.mark.parametrize("update_state", [update_as_an_unknown_node, update_an_unknown_channel])
def test_update_state_with_an_unknown_name_is_rejected(
    journal: Journal, update_state: Callable[[Thread[State, State, State]], object]
) -> None:
    thread = build_thread(build_chain(journal, ["node"]))
    run_until_done(thread.invoke({}))

    with pytest.raises(ValueError):
        update_state(thread)


def test_update_state_of_a_thread_without_a_checkpoint_is_rejected(journal: Journal) -> None:
    thread = build_thread(build_chain(journal, ["node"]))

    with pytest.raises(LookupError):
        thread.update_state({"items": ["edit"]})


@pytest.mark.parametrize("replayed_step", [0, 1, 2])
def test_replay_runs_the_tasks_of_an_earlier_checkpoint_again_on_a_new_branch(
    journal: Journal, replayed_step: int
) -> None:
    names = ["first", "second", "third"]
    thread = build_thread(build_chain(journal, names))
    expected = run_until_done(thread.invoke({}))
    replayed = next(snapshot for snapshot in thread.history() if snapshot.checkpoint.step == replayed_step)
    journal.events.clear()

    actual = run_until_done(thread.replay(replayed.checkpoint.id))
    fork = next(snapshot for snapshot in thread.history() if snapshot.checkpoint.source is CheckpointSource.fork)

    assert actual == expected, f"Expected the output {expected}, got {actual}."
    assert journal.events == names[replayed_step:], f"Expected the later nodes to run again, got {journal.events}."
    assert fork.checkpoint.parent_id == replayed.checkpoint.id, "Expected the fork to follow the replayed checkpoint."


def test_replay_of_an_interrupted_checkpoint_asks_again(journal: Journal) -> None:
    thread = build_single_asker(journal)
    first = read_interrupted(thread.invoke({}))
    run_until_done(thread.resume(Answer(interrupt_id=first.interrupts[0].interrupt_id, value="a")))
    interrupted_state = next(snapshot for snapshot in thread.history() if snapshot.checkpoint.step == 0)

    second = read_interrupted(thread.replay(interrupted_state.checkpoint.id))

    assert second.interrupts[0].interrupt_id != first.interrupts[0].interrupt_id, "Expected a new interrupt."
    assert second.interrupts[0].payload == first.interrupts[0].payload, "Expected the same question."


@pytest.mark.parametrize(
    ("input_schema", "payload"),
    [
        (DictRequest, {"topic": "cats"}),
        (DataclassRequest, DataclassRequest(topic="cats", tags=("a", "b"))),
        (ModelRequest, {"topic": "cats"}),
        (AliasRequest, AliasRequest(subject="cats")),
    ],
    ids=["typed-dict", "dataclass", "model", "model-with-an-alias"],
)
def test_send_payload_waiting_in_a_checkpoint_reaches_its_node_after_a_resume(
    journal: Journal, input_schema: type[object], payload: object
) -> None:
    received: list[object] = []
    failures: list[Exception] = [LookupError("fail one time")]

    def receive(request: object) -> State:
        if failures:
            raise failures.pop(0)
        received.append(request)
        return {}

    builder = StateGraph(State)
    builder.add_node("plan", lambda state: Command[str](goto=Send("receive", payload)))
    builder.add_node("receive", receive, input_schema=input_schema)
    builder.add_edge(Endpoint.start, "plan")
    thread = build_thread(builder)
    with pytest.raises(LookupError):
        thread.invoke({})
    expected = [input_schema(**payload)] if isinstance(payload, dict) else [payload]

    run_until_done(thread.resume())

    assert received == expected, f"Expected the node to receive {expected}, got {received}."


def send_a_hidden_count(state: object) -> Command[str]:
    request = HiddenRequest(topic="cats")
    request.count = 1
    return Command(goto=Send("receive", request))


def write_a_dog(state: object) -> PetState:
    return {"pet": Dog(name="rex", sound="woof")}


@pytest.mark.parametrize(("schema", "write"), [(PetState, write_a_dog), (State, send_a_hidden_count)])
def test_value_that_a_checkpoint_cannot_restore_is_rejected(
    schema: type[object], write: Callable[[object], object]
) -> None:
    builder = StateGraph[object, object, object](schema)
    builder.add_node("write", write)
    builder.add_node("receive", lambda request: None, input_schema=HiddenRequest)
    builder.add_edge(Endpoint.start, "write")
    thread = Thread(graph=builder.compile(), checkpointer=InMemoryCheckpointer(), thread_id="thread")

    with pytest.raises(ValueError):
        thread.invoke({})


@pytest.mark.parametrize("changed", ["node", "channel"])
def test_checkpoint_of_another_graph_is_rejected(journal: Journal, changed: str) -> None:
    thread = build_thread(build_chain(journal, ["ask"]))
    builder = StateGraph(State)
    builder.add_node("ask", build_asker(journal, "ask", 1))
    builder.add_edge(Endpoint.start, "ask")
    read_interrupted(build_thread(builder, thread.checkpointer).invoke({"topic": "cats"}))

    class OtherState(TypedDict, total=False):
        items: Annotated[list[str], Reducer(join_lists, initial_factory=list)]

    name = "ask" if changed == "channel" else "other"
    other = StateGraph(OtherState if changed == "channel" else State)
    other.add_node(name, lambda state: None)
    other.add_edge(Endpoint.start, name)

    with pytest.raises(ValueError):
        Thread(graph=other.compile(), checkpointer=thread.checkpointer, thread_id="thread").resume()


def test_state_of_an_unknown_checkpoint_is_none(journal: Journal) -> None:
    thread = build_thread(build_chain(journal, ["node"]))
    run_until_done(thread.invoke({}))
    run_until_done(Thread(graph=thread.graph, checkpointer=thread.checkpointer, thread_id="other").invoke({}))
    other_id = (
        Thread(graph=thread.graph, checkpointer=thread.checkpointer, thread_id="other").history()[0].checkpoint.id
    )

    assert thread.get_state(other_id) is None, "Expected no state for a checkpoint of another thread."


def test_state_of_a_child_checkpoint_is_rejected(journal: Journal) -> None:
    checkpointer = InMemoryCheckpointer()
    thread = build_thread(wrap_in_children(build_chain(journal, ["node"]), 1), checkpointer)
    run_until_done(thread.invoke({}))
    child_ids = [
        checkpoint_id
        for checkpoint_id in checkpointer.threads["thread"]
        if checkpoint_id not in {snapshot.checkpoint.id for snapshot in thread.history()}
    ]

    with pytest.raises(ValueError):
        thread.get_state(child_ids[0])


def test_delete_removes_the_checkpoints_of_the_thread(journal: Journal) -> None:
    thread = build_thread(build_chain(journal, ["node"]))
    run_until_done(thread.invoke({}))

    thread.delete()

    assert thread.history() == [], "Expected no checkpoint after the delete."
    assert thread.get_state() is None, "Expected no state after the delete."


@pytest.mark.parametrize("step_limit", [1, 2])
def test_step_limit_counts_the_steps_of_one_run(journal: Journal, step_limit: int) -> None:
    names = [f"node-{index}" for index in range(step_limit + 1)]
    thread = build_thread(build_chain(journal, names))

    with pytest.raises(RuntimeError):
        thread.invoke({}, step_limit=step_limit)
    output = run_until_done(thread.resume(step_limit=step_limit))

    assert output == {"items": names}, f"Expected the resumed run to finish, got {output}."


def test_async_thread_methods_give_the_results_of_the_sync_methods(journal: Journal) -> None:
    sync_thread = build_single_asker(journal)
    async_thread = build_single_asker(journal)

    async def run_async() -> tuple[object, ...]:
        interrupted = read_interrupted(await async_thread.ainvoke({}))
        answer = Answer(interrupt_id=interrupted.interrupts[0].interrupt_id, value="a")
        output = run_until_done(await async_thread.aresume(answer))
        first = (await async_thread.ahistory())[-1].checkpoint.id
        replayed = read_interrupted(await async_thread.areplay(first))
        update = await async_thread.aupdate_state({"items": ["edit"]})
        state = await async_thread.aget_state()
        await async_thread.adelete()
        return output, replayed.interrupts[0].payload, update == (state and state.checkpoint.id)

    interrupted = read_interrupted(sync_thread.invoke({}))
    output = run_until_done(sync_thread.resume(Answer(interrupt_id=interrupted.interrupts[0].interrupt_id, value="a")))
    replayed = read_interrupted(sync_thread.replay(sync_thread.history()[-1].checkpoint.id))
    update = sync_thread.update_state({"items": ["edit"]})
    state = sync_thread.get_state()
    expected = (output, replayed.interrupts[0].payload, update == (state and state.checkpoint.id))

    actual = asyncio.run(run_async())

    assert actual == expected, f"Expected {expected}, got {actual}."
    assert asyncio.run(async_thread.ahistory()) == [], "Expected adelete to remove the checkpoints."


def invoke_again(thread: Thread[State, State, State]) -> object:
    return thread.invoke({})


def resume_again(thread: Thread[State, State, State]) -> object:
    return thread.resume()


def replay_the_last_checkpoint(thread: Thread[State, State, State]) -> object:
    return thread.replay(thread.history()[0].checkpoint.id)


@pytest.mark.parametrize(
    ("call", "async_method"),
    [(invoke_again, "ainvoke"), (resume_again, "aresume"), (replay_the_last_checkpoint, "areplay")],
)
def test_sync_run_of_a_thread_rejects_a_graph_that_has_an_async_node(
    call: Callable[[Thread[State, State, State]], object], async_method: str
) -> None:
    async def write_async(state: State) -> State:
        return {}

    builder = StateGraph(State)
    builder.add_node("write", write_async)
    builder.add_edge(Endpoint.start, "write")
    thread = build_thread(builder)
    asyncio.run(thread.ainvoke({}))

    with pytest.raises(TypeError, match=async_method):
        call(thread)


def build_sibling_graph(journal: Journal, sibling: Callable[[State], State]) -> Builder:
    builder = StateGraph(State)
    builder.add_node("ask", build_asker(journal, "ask", 1))
    builder.add_node("sibling", sibling)
    builder.add_node("after_ask", build_writer(journal, "after_ask"))
    builder.add_node("after_sibling", build_writer(journal, "after_sibling"))
    builder.add_edge(Endpoint.start, "ask")
    builder.add_edge(Endpoint.start, "sibling")
    builder.add_edge("ask", "after_ask")
    builder.add_edge("sibling", "after_sibling")
    return builder


def test_update_as_a_node_keeps_the_outcomes_of_its_finished_siblings(journal: Journal) -> None:
    thread = build_thread(build_sibling_graph(journal, build_writer(journal, "sibling")))
    read_interrupted(thread.invoke({}))

    thread.update_state({"items": ["edit"]}, as_node="ask")
    state = thread.get_state()
    output = run_until_done(thread.resume())

    assert state is not None, "Expected the saved state."
    assert state.next == ("after_ask", "after_sibling"), f"Expected the successors of both nodes, got {state.next}."
    assert output == {"items": ["sibling", "edit", "after_ask", "after_sibling"]}, f"Expected all writes, got {output}."
    assert sorted(journal.events) == ["after_ask", "after_sibling", "ask", "sibling"], (
        f"Expected the finished sibling not to run again, got {journal.events}."
    )


def test_update_as_a_node_while_another_task_waits_is_rejected(journal: Journal) -> None:
    thread = build_thread(build_sibling_graph(journal, build_asker(journal, "sibling", 1)))
    read_interrupted(thread.invoke({}))

    with pytest.raises(ValueError):
        thread.update_state({"items": ["edit"]}, as_node="ask")


def test_update_without_a_node_runs_every_planned_task_again(journal: Journal) -> None:
    thread = build_thread(build_sibling_graph(journal, build_writer(journal, "sibling")))
    read_interrupted(thread.invoke({}))

    thread.update_state({"topic": "edited"})
    state = thread.get_state()

    assert state is not None, "Expected the saved state."
    assert state.next == ("ask", "sibling"), f"Expected both planned tasks to run again, got {state.next}."
    assert state.interrupts == (), f"Expected the old interrupt to be gone, got {state.interrupts}."


class Point(BaseModel):
    x: int
    y: int


class CoercedState(TypedDict, total=False):
    items: list[int]
    point: Point
    count: int


@pytest.mark.parametrize(
    "write",
    [{"items": (1, 2)}, {"point": {"x": 1, "y": 2}}, {"count": "3"}],
    ids=["tuple-into-a-list", "dict-into-a-model", "string-into-an-int"],
)
def test_thread_accepts_the_writes_that_a_plain_run_parses(write: dict[str, object]) -> None:
    builder = StateGraph(CoercedState)
    builder.add_node("write", lambda state: write)
    builder.add_edge(Endpoint.start, "write")
    graph = builder.compile()
    expected = Finished(graph.invoke({}))

    actual = Thread(graph=graph, checkpointer=InMemoryCheckpointer(), thread_id="thread").invoke({})

    assert actual == expected, f"Expected {expected}, got {actual}."


class LimitState(TypedDict, total=False):
    name: str
    limit: int


class LimitRequest(BaseModel):
    name: str
    limit: int = 10


def test_send_payload_keeps_its_unset_fields_unset_after_a_resume() -> None:
    failures: list[Exception] = [LookupError("fail one time")]

    def review(request: LimitRequest) -> LimitRequest:
        if failures:
            raise failures.pop(0)
        return request

    builder = StateGraph(LimitState)
    builder.add_node("plan", lambda state: Command[str](goto=Send("review", LimitRequest(name="cats"))))
    builder.add_node("review", review, input_schema=LimitRequest)
    builder.add_edge(Endpoint.start, "plan")
    graph = builder.compile()
    thread = Thread(graph=graph, checkpointer=InMemoryCheckpointer(), thread_id="thread")
    with pytest.raises(LookupError):
        thread.invoke({"limit": 5})

    output = run_until_done(thread.resume())

    assert output == {"name": "cats", "limit": 5}, f"Expected the unset limit not to be written, got {output}."


@pytest.mark.parametrize("depth", [1, 2])
def test_interrupt_inside_task_groups_of_an_async_node_stops_the_run(depth: int) -> None:
    async def ask(levels: int) -> str:
        if levels == 0:
            await asyncio.sleep(0)
            return interrupt("topic?", str)
        async with asyncio.TaskGroup() as group:
            answer = group.create_task(ask(levels - 1))
        return answer.result()

    async def ask_in_groups(state: State) -> State:
        return {"items": [await ask(depth)]}

    builder = StateGraph(State)
    builder.add_node("ask", ask_in_groups)
    builder.add_edge(Endpoint.start, "ask")
    thread = build_thread(builder)

    async def ask_and_answer() -> RunResult[State]:
        interrupted = read_interrupted(await thread.ainvoke({}))
        return await thread.aresume(Answer(interrupt_id=interrupted.interrupts[0].interrupt_id, value="cats"))

    output = run_until_done(asyncio.run(ask_and_answer()))

    assert output == {"items": ["cats"]}, f"Expected the answer in the output, got {output}."


def test_exception_group_of_a_node_without_an_interrupt_leaves_the_run_unchanged() -> None:
    failure = ExceptionGroup("node failures", [ValueError("wrong value")])

    def fail(state: State) -> State:
        raise failure

    builder = StateGraph(State)
    builder.add_node("fail", fail)
    builder.add_edge(Endpoint.start, "fail")

    with pytest.raises(ExceptionGroup) as raised:
        build_thread(builder).invoke({})
    assert raised.value is failure, f"Expected {failure!r}, got {raised.value!r}. The group must not be wrapped."


def test_update_that_the_reducer_does_not_take_is_rejected() -> None:
    builder = StateGraph(TopicState)
    builder.add_node("write", lambda state: {"topics": "not a topic"})
    builder.add_edge(Endpoint.start, "write")
    thread = Thread(graph=builder.compile(), checkpointer=InMemoryCheckpointer(), thread_id="thread")

    with pytest.raises(ValueError):
        thread.invoke({})


@dataclass
class SlowCheckpointer(InMemoryCheckpointer):
    recording: asyncio.Event = field(default_factory=asyncio.Event)

    async def put_task_state(self, thread_id: str, task: TaskAddress, state: TaskState) -> None:
        self.recording.set()
        await asyncio.sleep(0.01)
        await super().put_task_state(thread_id, task, state)


def test_finished_task_is_recorded_when_a_sibling_fails_during_the_record(journal: Journal) -> None:
    checkpointer = SlowCheckpointer()
    failures: list[Exception] = [LookupError("fail one time")]

    async def charge(state: State) -> State:
        journal.events.append("charge")
        return {"items": ["charge"]}

    async def fail_while_recording(state: State) -> State:
        await checkpointer.recording.wait()
        if failures:
            raise failures.pop(0)
        return {"items": ["flaky"]}

    builder = StateGraph(State)
    builder.add_node("charge", charge)
    builder.add_node("flaky", fail_while_recording)
    builder.add_edge(Endpoint.start, "charge")
    builder.add_edge(Endpoint.start, "flaky")
    thread = build_thread(builder, checkpointer)

    async def fail_and_resume() -> RunResult[State]:
        with pytest.raises(LookupError):
            await thread.ainvoke({})
        return await thread.aresume()

    output = run_until_done(asyncio.run(fail_and_resume()))

    assert journal.events == ["charge"], f"Expected the finished node not to run again, got {journal.events}."
    assert sorted(output.get("items", [])) == ["charge", "flaky"], f"Expected both writes, got {output}."

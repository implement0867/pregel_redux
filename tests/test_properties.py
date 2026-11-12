import asyncio
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, TypedDict, assert_never

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine

from pregel_redux import (
    Checkpointer,
    Command,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupted,
    Reducer,
    Send,
    StateGraph,
    ThreadConfig,
    interrupt,
)
from pregel_redux.sqlite import SqliteCheckpointer
from tests.conftest import collect, join_lists


class State(TypedDict, total=False):
    origin: str
    visits: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class Origin(TypedDict, total=False):
    origin: str


class Visits(TypedDict, total=False):
    visits: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


@dataclass(frozen=True)
class WriterSpec:
    pass


@dataclass(frozen=True)
class AskerSpec:
    questions: int


@dataclass(frozen=True)
class RouterSpec:
    forward: tuple[int, ...]
    sends: tuple[int, ...]
    back: tuple[int, ...]
    rounds: int


@dataclass(frozen=True)
class FunctionSpec:
    action: WriterSpec | AskerSpec | RouterSpec
    is_async: bool
    fails: bool


@dataclass(frozen=True)
class ChildSpec:
    graph: "GraphSpec"


@dataclass(frozen=True)
class NodeSpec:
    name: str
    work: FunctionSpec | ChildSpec
    defer: bool
    targets: tuple[int, ...]
    to_end: bool


@dataclass(frozen=True)
class JoinSpec:
    sources: tuple[int, ...]
    target: int | None


@dataclass(frozen=True)
class GraphSpec:
    nodes: tuple[NodeSpec, ...]
    entries: tuple[int, ...]
    joins: tuple[JoinSpec, ...]


@dataclass(frozen=True)
class Build:
    interactive: bool
    async_nodes: bool
    failed: set[tuple[int, ...]] | None


def pick_some(items: Sequence[int]) -> st.SearchStrategy[tuple[int, ...]]:
    if not items:
        return st.just(())
    return st.lists(st.sampled_from(items), unique=True).map(tuple)


@st.composite
def join_specs(draw: st.DrawFn, node_count: int) -> JoinSpec:
    target = draw(st.one_of(st.none(), st.integers(min_value=2, max_value=node_count - 1)))
    sources = draw(st.lists(st.sampled_from(range(node_count if target is None else target)), min_size=2, unique=True))
    return JoinSpec(tuple(sorted(sources)), target)


@st.composite
def graph_specs(draw: st.DrawFn, known_names: tuple[str, ...], depth: int) -> GraphSpec:
    name = st.one_of(st.sampled_from(known_names), st.text()) if known_names else st.text()
    names = list(dict.fromkeys(draw(st.lists(name, min_size=1, max_size=5))))
    nodes: list[NodeSpec] = []
    for index, node_name in enumerate(names):
        later = range(index + 1, len(names))
        action = st.one_of(
            st.just(WriterSpec()),
            st.builds(AskerSpec, questions=st.integers(min_value=1, max_value=3)),
            st.builds(
                RouterSpec,
                forward=pick_some(later),
                sends=pick_some(later),
                back=pick_some(range(index + 1)),
                rounds=st.integers(min_value=0, max_value=2),
            ),
        )
        function = st.builds(
            FunctionSpec,
            action=action,
            is_async=st.booleans(),
            fails=st.booleans(),
        )
        child = st.builds(ChildSpec, graph=graph_specs((*known_names, *names), depth + 1))
        work = draw(function if depth == 2 else st.one_of(function, child))
        nodes.append(NodeSpec(node_name, work, draw(st.booleans()), draw(pick_some(later)), draw(st.booleans())))
    entries = draw(st.lists(st.sampled_from(range(len(names))), min_size=1, unique=True))
    joins = draw(st.lists(join_specs(len(names)))) if len(names) > 2 else []
    return GraphSpec(tuple(nodes), tuple(entries), tuple(joins))


def build_function(
    spec: GraphSpec, node: NodeSpec, work: FunctionSpec, path: tuple[int, ...], build: Build
) -> Callable[[State], object]:
    name = node.name

    def send(index: int, visits: list[str]) -> Send:
        target = spec.nodes[index]
        match target.work:
            case FunctionSpec():
                return Send(target.name, State(visits=visits))
            case ChildSpec():
                return Send(target.name, Origin(origin=name))
            case _:
                assert_never(target.work)

    def act(state: State) -> object:
        visits = state.get("visits", [])
        label = f"{name}|{state.get('origin')}|{len(visits)}"
        result: object
        match work.action:
            case WriterSpec():
                result = State(visits=[label])
            case AskerSpec(questions=questions):
                asked = [f"{name} {index}" for index in range(questions)]
                answers = [
                    interrupt(question, response_schema=str) if build.interactive else f"answer to {question}"
                    for question in asked
                ]
                result = State(visits=[label, *answers])
            case RouterSpec() as router:
                back = router.back if sum(visit.startswith(f"{name}|") for visit in visits) < router.rounds else ()
                result = Command[str](
                    update=State(visits=[label]),
                    goto=[
                        *(spec.nodes[index].name for index in (*router.forward, *back)),
                        *(send(index, visits) for index in router.sends),
                    ],
                )
            case _:
                assert_never(work.action)
        if work.fails and build.failed is not None and path not in build.failed:
            build.failed.add(path)
            raise ConnectionError(name)
        return result

    async def act_async(state: State) -> object:
        return act(state)

    return act_async if work.is_async and build.async_nodes else act


def add_nodes[InputT, OutputT](
    builder: StateGraph[State, InputT, OutputT], spec: GraphSpec, path: tuple[int, ...], build: Build
) -> None:
    names = [node.name for node in spec.nodes]
    for index, node in enumerate(spec.nodes):
        match node.work:
            case FunctionSpec() as work:
                builder.add_node(node.name, build_function(spec, node, work, (*path, index), build), defer=node.defer)
            case ChildSpec(graph=graph):
                child = StateGraph(State, input_schema=Origin, output_schema=Visits)
                add_nodes(child, graph, (*path, index), build)
                builder.add_node(node.name, child, defer=node.defer)
            case _:
                assert_never(node.work)
        for target in node.targets:
            builder.add_edge(node.name, names[target])
        if node.to_end:
            builder.add_edge(node.name, Endpoint.end)
    for entry in spec.entries:
        builder.add_edge(Endpoint.start, names[entry])
    for join in spec.joins:
        first, second, *others = (names[source] for source in join.sources)
        builder.add_edge((first, second, *others), Endpoint.end if join.target is None else names[join.target])


def build_graph(spec: GraphSpec, build: Build) -> StateGraph[State, State, State]:
    builder = StateGraph(State)
    add_nodes(builder, spec, (), build)
    return builder


def run_to_the_end(invoke: Callable[[State | Command[str] | None], Finished[State] | Interrupted]) -> State:
    def attempt(run_input: State | Command[str] | None) -> Finished[State] | Interrupted | None:
        result: Finished[State] | Interrupted | None = None
        with suppress(ConnectionError):
            result = invoke(run_input)
        return result

    next_input: State | Command[str] | None = State(origin="input")
    while True:
        match attempt(next_input):
            case Finished(output=output):
                return output
            case None | Interrupted(interrupts=()):
                next_input = None
            case Interrupted(interrupts=interrupts):
                next_input = Command(resume_by_interrupt_id={item.id: f"answer to {item.value}" for item in interrupts})
            case _ as unknown:
                assert_never(unknown)


@contextmanager
def open_memory() -> Generator[Checkpointer]:
    yield InMemoryCheckpointer()


@contextmanager
def open_sqlite() -> Generator[Checkpointer]:
    with TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{Path(directory) / 'checkpoints.sqlite'}")
        checkpointer = SqliteCheckpointer(engine)
        checkpointer.setup()
        yield checkpointer
        engine.dispose()


def walk_graphs(spec: GraphSpec) -> Iterator[GraphSpec]:
    yield spec
    for node in spec.nodes:
        match node.work:
            case ChildSpec(graph=graph):
                yield from walk_graphs(graph)
            case FunctionSpec():
                pass
            case _:
                assert_never(node.work)


def record_features(spec: GraphSpec) -> None:
    graphs = list(walk_graphs(spec))
    nodes = [node for graph in graphs for node in graph.nodes]
    routers = [
        node.work.action
        for node in nodes
        if isinstance(node.work, FunctionSpec) and isinstance(node.work.action, RouterSpec)
    ]
    child_counts = [sum(isinstance(node.work, ChildSpec) for node in graph.nodes) for graph in graphs]
    child_names = {node.name for graph in graphs[1:] for node in graph.nodes}
    event(f"graph with three or more children: {max(child_counts) > 2}")
    event(
        f"levels: {1 + max(child_counts[0] and 1, *(2 for graph in graphs[1:] if any(isinstance(n.work, ChildSpec) for n in graph.nodes)), 0)}"
    )
    event(f"name collision between levels: {bool({node.name for node in spec.nodes} & child_names)}")
    event(f"join: {any(graph.joins for graph in graphs)}")
    event(f"send: {any(router.sends for router in routers)}")
    event(f"route back: {any(router.back and router.rounds for router in routers)}")
    event(
        f"interrupt: {any(isinstance(node.work, FunctionSpec) and isinstance(node.work.action, AskerSpec) for node in nodes)}"
    )
    event(f"deferred node: {any(node.defer for node in nodes)}")


@pytest.mark.parametrize("open_checkpointer", [open_memory, open_sqlite])
@settings(deadline=None)
@given(spec=graph_specs((), 0), interactive=st.booleans(), failing=st.booleans(), data=st.data())
def test_run_with_a_checkpointer_gives_the_output_of_a_run_without_one(
    open_checkpointer: Callable[[], AbstractContextManager[Checkpointer]],
    spec: GraphSpec,
    data: st.DataObject,
    *,
    interactive: bool,
    failing: bool,
) -> None:
    """
    Arrange: a generated graph, with and without a checkpointer, whose nodes can fail one time and can ask questions.
    Act: run the graph without a checkpointer, and run it with the checkpointer until it finishes.
    Assert: the two runs give the same output.
    """
    record_features(spec)
    names = [node.name for node in spec.nodes]
    interrupt_before = data.draw(st.lists(st.sampled_from(names), unique=True))
    interrupt_after = data.draw(st.lists(st.sampled_from(names), unique=True))
    plain = build_graph(spec, Build(interactive=False, async_nodes=True, failed=None)).compile()

    with open_checkpointer() as checkpointer, asyncio.Runner() as runner:
        expected = runner.run(plain.ainvoke(State(origin="input"), step_limit=1000))
        graph = build_graph(
            spec, Build(interactive=interactive, async_nodes=True, failed=set() if failing else None)
        ).compile(checkpointer=checkpointer, interrupt_before=interrupt_before, interrupt_after=interrupt_after)
        actual = run_to_the_end(
            lambda run_input: runner.run(graph.ainvoke(run_input, {"thread_id": "thread"}, step_limit=1000))
        )

    assert actual == expected, f"Expected {expected}, got {actual}, for {spec}."


@pytest.mark.parametrize("open_checkpointer", [open_memory, open_sqlite])
@settings(deadline=None)
@given(spec=graph_specs((), 0))
def test_sync_run_gives_the_output_and_the_history_of_an_async_run(
    open_checkpointer: Callable[[], AbstractContextManager[Checkpointer]], spec: GraphSpec
) -> None:
    """
    Arrange: a generated graph of sync nodes that can fail one time and can ask questions.
    Act: run the graph to its end with the async function and with the sync function, each in its own thread.
    Assert: the two runs give the same output, and their histories have the same steps and values.
    """
    with open_checkpointer() as checkpointer, asyncio.Runner() as runner:
        graph = build_graph(spec, Build(interactive=True, async_nodes=True, failed=set())).compile(
            checkpointer=checkpointer
        )
        expected = run_to_the_end(
            lambda run_input: runner.run(graph.ainvoke(run_input, {"thread_id": "async"}, step_limit=1000))
        )
        sync_graph = build_graph(spec, Build(interactive=True, async_nodes=False, failed=set())).compile(
            checkpointer=checkpointer
        )

        actual = run_to_the_end(lambda run_input: sync_graph.invoke(run_input, {"thread_id": "sync"}, step_limit=1000))

        histories = [
            [(snapshot.step, snapshot.values, snapshot.next) for snapshot in graph.get_state_history(config)]
            for config in (ThreadConfig(thread_id="async"), ThreadConfig(thread_id="sync"))
        ]
    assert actual == expected, f"Expected {expected}, got {actual}, for {spec}."
    assert histories[1] == histories[0], f"Expected the history {histories[0]}, got {histories[1]}, for {spec}."


@pytest.mark.parametrize("open_checkpointer", [open_memory, open_sqlite])
@settings(deadline=None)
@given(spec=graph_specs((), 0))
def test_history_of_a_finished_run_is_a_chain_of_its_checkpoints(
    open_checkpointer: Callable[[], AbstractContextManager[Checkpointer]], spec: GraphSpec
) -> None:
    """
    Arrange: a generated graph that ran to its end, with interrupts and failures.
    Act: read the history of the thread, and read each state of the history by its checkpoint id.
    Assert: the newest state has the output, each state has the next older state as parent, and each read is equal.
    """
    config = ThreadConfig(thread_id="thread")
    with open_checkpointer() as checkpointer, asyncio.Runner() as runner:
        graph = build_graph(spec, Build(interactive=True, async_nodes=True, failed=set())).compile(
            checkpointer=checkpointer
        )
        output = run_to_the_end(lambda run_input: runner.run(graph.ainvoke(run_input, config, step_limit=1000)))

        history = runner.run(collect(graph.aget_state_history(config)))
        found = [
            runner.run(graph.aget_state(ThreadConfig(thread_id="thread", checkpoint_id=snapshot.checkpoint_id)))
            for snapshot in history
        ]

    steps = [snapshot.step for snapshot in history]
    parents = [snapshot.parent_id for snapshot in history]
    expected_parents = [*(snapshot.checkpoint_id for snapshot in history[1:]), None]
    expected_steps = list(range(len(history) - 2, -2, -1))
    assert history[0].values == output, f"Expected the newest state {output}, got {history[0].values}."
    assert history[0].next == (), f"Expected no next node after the end, got {history[0].next}."
    assert steps == expected_steps, f"Expected the steps {expected_steps}, got {steps}."
    assert parents == expected_parents, f"Expected the parents {expected_parents}, got {parents}."
    assert found == history, "Expected each state read by its checkpoint id to be equal to the state of the history."

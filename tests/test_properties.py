import asyncio
import json
import threading
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, aclosing, closing, contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, TypedDict, assert_never
from uuid import UUID

import pytest
from hypothesis import event, find, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule
from pydantic import TypeAdapter
from sqlalchemy import create_engine

from pregel_redux import (
    Checkpointer,
    Command,
    Depends,
    Endpoint,
    Finished,
    InMemoryCheckpointer,
    Interrupt,
    Interrupted,
    PersistentGraph,
    Reducer,
    Scope,
    Send,
    StateGraph,
    StateSnapshot,
    StreamMode,
    ThreadConfig,
    interrupt,
)
from pregel_redux.checkpoints import CheckpointSource
from pregel_redux.snapshots import PullPath, TaskSnapshot
from pregel_redux.sqlite import SqliteCheckpointer
from pregel_redux.streams import StreamPart, ValuesPart
from tests.conftest import collect, join_lists


def add_numbers(current: int, update: int) -> int:
    return current + update


class State(TypedDict, total=False):
    origin: str
    clock: Annotated[int, Reducer(add_numbers, initial_factory=int)]
    visits: Annotated[list[str], Reducer(join_lists, initial_factory=list)]


class ChildInput(TypedDict, total=False):
    origin: str
    clock: Annotated[int, Reducer(add_numbers, initial_factory=int)]


class ChildOutput(TypedDict, total=False):
    clock: Annotated[int, Reducer(add_numbers, initial_factory=int)]
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
class ResourceSpec:
    scope: Scope
    is_async: bool


@dataclass(frozen=True)
class FunctionSpec:
    action: WriterSpec | AskerSpec | RouterSpec
    is_async: bool
    pauses: int
    late_pauses: int
    resource: ResourceSpec | None


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


class Phase(Enum):
    start = "start"
    end = "end"
    cleanup = "cleanup"


@dataclass
class Execution:
    path: tuple[int, ...]
    era: int
    origin: str | None
    clock: int
    observed: tuple[str, ...]
    given: list[str | None] = field(default_factory=list)
    received: list[str | None] = field(default_factory=list)
    completed: bool = False


@dataclass
class Trace:
    executions: dict[str, Execution] = field(default_factory=dict)
    tags: dict[str, tuple[str, int]] = field(default_factory=dict)
    faults: dict[tuple[int, ...], Phase] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    given: dict[str, str] = field(default_factory=dict)
    plain: str | None = None
    fired: list[tuple[tuple[int, ...], Phase]] = field(default_factory=list)
    era: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin(self, execution: Execution) -> str:
        with self.lock:
            label = f"#{len(self.executions)}"
            self.executions[label] = execution
            return label

    def fail(self, path: tuple[int, ...], phase: Phase) -> None:
        with self.lock:
            armed = self.faults.get(path) is phase
            if armed:
                del self.faults[path]
                self.fired.append((path, phase))
        if armed:
            raise ConnectionError(path)

    def start_era(self) -> None:
        self.era += 1
        self.consumed.clear()

    def completed(self) -> list[Execution]:
        return [execution for execution in list(self.executions.values()) if execution.completed]


@dataclass(frozen=True)
class Build:
    interactive: bool
    async_nodes: bool
    trace: Trace


def pick_some(items: Sequence[int]) -> st.SearchStrategy[tuple[int, ...]]:
    if not items:
        return st.just(())
    return st.lists(st.sampled_from(items), unique=True).map(tuple)


@st.composite
def join_specs(draw: st.DrawFn, node_count: int, reached: Sequence[int]) -> JoinSpec:
    target = draw(st.one_of(st.none(), st.integers(min_value=2, max_value=node_count - 1)))
    limit = node_count if target is None else target
    live = [index for index in reached if index < limit]
    any_sources = st.lists(st.sampled_from(range(limit)), min_size=2, unique=True)
    live_sources = st.lists(st.sampled_from(live), min_size=2, unique=True) if len(live) > 1 else any_sources
    return JoinSpec(tuple(sorted(draw(st.one_of(live_sources, any_sources)))), target)


def drop_ways(node: NodeSpec, alone: set[int]) -> NodeSpec:
    def keep(indexes: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(index for index in indexes if index not in alone)

    work = node.work
    if isinstance(work, FunctionSpec) and isinstance(work.action, RouterSpec):
        action = RouterSpec(
            keep(work.action.forward), keep(work.action.sends), keep(work.action.back), work.action.rounds
        )
        work = replace(work, action=action)
    return replace(node, work=work, targets=keep(node.targets))


def find_reached_nodes(nodes: Sequence[NodeSpec], entries: Sequence[int]) -> list[int]:
    reached = set(entries)
    for index, node in enumerate(nodes):
        if index in reached:
            reached.update(node.targets)
            if isinstance(node.work, FunctionSpec) and isinstance(node.work.action, RouterSpec):
                reached.update((*node.work.action.forward, *node.work.action.sends))
    return sorted(reached)


@st.composite
def graph_specs(draw: st.DrawFn, known_names: tuple[str, ...], depth: int) -> GraphSpec:
    text = st.text(st.characters(exclude_categories=["Cs"], exclude_characters="|:")).filter(
        lambda name: name != "__start__"
    )
    name = st.one_of(st.sampled_from(known_names), text) if known_names else text
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
            pauses=st.integers(min_value=0, max_value=3),
            late_pauses=st.integers(min_value=0, max_value=3),
            resource=st.one_of(
                st.none(), st.builds(ResourceSpec, scope=st.sampled_from(Scope), is_async=st.booleans())
            ),
        )
        child = st.builds(ChildSpec, graph=graph_specs((*known_names, *names), depth + 1))
        work = draw(function if depth == 2 else st.one_of(function, child))
        nodes.append(NodeSpec(node_name, work, draw(st.booleans()), draw(pick_some(later)), draw(st.booleans())))
    entries = draw(st.lists(st.sampled_from(range(len(names))), min_size=1, unique=True))
    reached = find_reached_nodes(nodes, entries)
    joins = draw(st.lists(join_specs(len(names), reached))) if len(names) > 2 else []
    targets = sorted({join.target for join in joins if join.target is not None})
    alone = set(draw(pick_some(targets)))
    kept_entries = [entry for entry in entries if entry not in alone] or entries
    return GraphSpec(tuple(drop_ways(node, alone) for node in nodes), tuple(kept_entries), tuple(joins))


def build_function(spec: GraphSpec, work: FunctionSpec, path: tuple[int, ...], build: Build) -> Callable[..., object]:
    trace = build.trace

    def send(position: int, index: int, label: str, execution: Execution) -> Send:
        tag = f"{label}>{position}"
        trace.tags[tag] = (label, position)
        target = spec.nodes[index]
        match target.work:
            case FunctionSpec():
                return Send(
                    target.name, State(origin=tag, visits=[*execution.observed, label, tag], clock=execution.clock)
                )
            case ChildSpec():
                return Send(target.name, ChildInput(origin=tag, clock=execution.clock))
            case _:
                assert_never(target.work)

    def start(state: State) -> tuple[Execution, str]:
        execution = Execution(
            path, trace.era, state.get("origin"), state.get("clock", 0), tuple(state.get("visits", []))
        )
        label = trace.begin(execution)
        trace.fail(path, Phase.start)
        return execution, label

    def finish(execution: Execution, label: str) -> object:
        update = State(visits=[label], clock=1)
        result: object = update
        match work.action:
            case WriterSpec():
                pass
            case AskerSpec(questions=questions):
                for index in range(questions if build.interactive else 0):
                    question = json.dumps([list(path), execution.origin, execution.clock, index])
                    answer = interrupt(question, response_schema=str)
                    with trace.lock:
                        if answer == trace.plain and question not in trace.consumed:
                            trace.given[question] = answer
                            trace.plain = None
                    execution.given.append(trace.given.get(question))
                    execution.received.append(answer)
                    trace.consumed.add(question)
            case RouterSpec() as router:
                rounds = sum(
                    visit in trace.executions and trace.executions[visit].path == path for visit in execution.observed
                )
                back = router.back if rounds < router.rounds else ()
                result = Command[str](
                    update=update,
                    goto=[
                        *(spec.nodes[index].name for index in (*router.forward, *back)),
                        *(send(position, index, label, execution) for position, index in enumerate(router.sends)),
                    ],
                )
            case _:
                assert_never(work.action)
        trace.fail(path, Phase.end)
        execution.completed = True
        return result

    def act(state: State) -> object:
        return finish(*start(state))

    async def act_async(state: State) -> object:
        for _ in range(work.pauses):
            await asyncio.sleep(0)
        started = start(state)
        for _ in range(work.late_pauses):
            await asyncio.sleep(0)
        return finish(*started)

    is_async = work.is_async and build.async_nodes
    if work.resource is None:
        return act_async if is_async else act

    def open_resource() -> Iterator[int]:
        try:
            yield 0
        finally:
            trace.fail(path, Phase.cleanup)

    async def open_resource_async() -> AsyncIterator[int]:
        try:
            yield 0
        finally:
            trace.fail(path, Phase.cleanup)

    provider = open_resource_async if work.resource.is_async and build.async_nodes else open_resource
    scope = work.resource.scope

    def act_with_resource(state: State, resource: Annotated[int, Depends(provider, scope=scope)]) -> object:
        return act(state)

    async def act_async_with_resource(state: State, resource: Annotated[int, Depends(provider, scope=scope)]) -> object:
        return await act_async(state)

    return act_async_with_resource if is_async else act_with_resource


def add_nodes[InputT, OutputT](
    builder: StateGraph[State, InputT, OutputT], spec: GraphSpec, path: tuple[int, ...], build: Build
) -> None:
    names = [node.name for node in spec.nodes]
    for index, node in enumerate(spec.nodes):
        match node.work:
            case FunctionSpec() as work:
                builder.add_node(node.name, build_function(spec, work, (*path, index), build), defer=node.defer)
            case ChildSpec(graph=graph):
                child = StateGraph(State, input_schema=ChildInput, output_schema=ChildOutput)
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


def walk_graphs(spec: GraphSpec, path: tuple[int, ...]) -> Iterator[tuple[tuple[int, ...], GraphSpec]]:
    yield path, spec
    for index, node in enumerate(spec.nodes):
        match node.work:
            case ChildSpec(graph=graph):
                yield from walk_graphs(graph, (*path, index))
            case FunctionSpec():
                pass
            case _:
                assert_never(node.work)


def find_function_paths(spec: GraphSpec) -> list[tuple[int, ...]]:
    return [
        (*path, index)
        for path, graph in walk_graphs(spec, ())
        for index, node in enumerate(graph.nodes)
        if isinstance(node.work, FunctionSpec)
    ]


def has_async_function(spec: GraphSpec) -> bool:
    return any(
        isinstance(node.work, FunctionSpec) and node.work.is_async
        for _, graph in walk_graphs(spec, ())
        for node in graph.nodes
    )


def find_spec(spec: GraphSpec, checkpoint_ns: str) -> GraphSpec:
    found = spec
    for part in checkpoint_ns.split("|") if checkpoint_ns else ():
        name = part.rsplit(":", 1)[0]
        work = next(node.work for node in found.nodes if node.name == name)
        if not isinstance(work, ChildSpec):
            raise TypeError(f"Expected the namespace {checkpoint_ns!r} to name child graphs, got {name!r}.")
        found = work.graph
    return found


def canonize(trace: Trace, visits: Sequence[str]) -> tuple[list[object], Counter[object]]:
    names: dict[str, object] = {}

    def name(label: str) -> object:
        if label in names:
            return names[label]
        if label in trace.tags:
            sender, position = trace.tags[label]
            return ("send", name(sender), position)
        return ("unknown", label)

    for label, execution in list(trace.executions.items()):
        origin = None if execution.origin is None else name(execution.origin)
        names[label] = hash((execution.path, origin, execution.clock, tuple(name(seen) for seen in execution.observed)))
    return [name(visit) for visit in visits], Counter(
        names[label] for label, execution in trace.executions.items() if execution.completed
    )


def find_unfired_joins(spec: GraphSpec, trace: Trace) -> list[JoinSpec]:
    current = [
        (order, execution.path[0])
        for order, execution in enumerate(list(trace.executions.values()))
        if execution.completed and execution.era == trace.era
    ]
    unfired: list[JoinSpec] = []
    for join in spec.joins:
        last_run = max((order for order, node in current if node == join.target), default=-1)
        arrived = {node for order, node in current if order > last_run}
        if join.target is not None and arrived >= set(join.sources):
            unfired.append(join)
    return unfired


def run_plain(spec: GraphSpec, origin: str) -> tuple[Trace, State]:
    trace = Trace()
    graph = build_graph(spec, Build(interactive=False, async_nodes=True, trace=trace)).compile()
    return trace, asyncio.run(graph.ainvoke(State(origin=origin), step_limit=1000))


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


class CallMode(Enum):
    asynchronous = "asynchronous"
    sync = "sync"
    sync_in_loop = "sync_in_loop"


@dataclass(frozen=True)
class StreamPlan:
    modes: frozenset[StreamMode]
    subgraphs: bool
    stop_after: int | None


@dataclass(frozen=True)
class Delivery:
    mode: CallMode
    stream: StreamPlan | None


def deliveries() -> st.SearchStrategy[Delivery]:
    plan = st.builds(
        StreamPlan,
        modes=st.frozensets(st.sampled_from(StreamMode)),
        subgraphs=st.booleans(),
        stop_after=st.one_of(st.none(), st.integers(min_value=0)),
    )
    return st.builds(Delivery, mode=st.sampled_from(CallMode), stream=st.one_of(st.none(), plan))


async def take(parts: AsyncGenerator[StreamPart], count: int | None) -> list[StreamPart]:
    taken: list[StreamPart] = []
    async with aclosing(parts):
        async for part in parts:
            if len(taken) == count:
                break
            taken.append(part)
    return taken


async def call_in_loop[ResultT](function: Callable[[], ResultT]) -> ResultT:
    return function()


def read_question(waiting: Interrupt) -> str:
    match waiting.value:
        case str() as question:
            return question
        case _:
            raise AssertionError(f"Expected a question text as the value of the interrupt, got {waiting.value!r}.")


def read_visits(values: Mapping[str, object]) -> list[str]:
    return TypeAdapter(list[str]).validate_python(values.get("visits", []))


def find_errors(error: BaseException) -> list[BaseException]:
    found: dict[int, BaseException] = {}
    pending: list[BaseException | None] = [error]
    while pending:
        match pending.pop():
            case None:
                pass
            case BaseExceptionGroup(exceptions=members) as group:
                found[id(group)] = group
                pending.extend(member for member in members if id(member) not in found)
            case item if id(item) not in found:
                found[id(item)] = item
                pending.extend((item.__cause__, item.__context__))
            case _:
                pass
    return list(found.values())


def find_namespaces(snapshot: StateSnapshot) -> Iterator[StateSnapshot]:
    for task in snapshot.tasks:
        if task.state is not None:
            yield task.state
            yield from find_namespaces(task.state)


@dataclass
class Thread:
    spec: GraphSpec
    graph: PersistentGraph[State, State, State]
    trace: Trace
    origin: str
    expected: tuple[list[object], Counter[object]]
    sync_allowed: bool
    interrupt_before: frozenset[str]
    interrupt_after: frozenset[str]
    promised: list[str] = field(default_factory=list)
    serial: int = 0
    diverged: bool = False
    steered: bool = False
    result: Finished[State] | Interrupted | None = None
    snapshot: StateSnapshot | None = None
    history: list[StateSnapshot] = field(default_factory=list)

    def make_text(self, text: str) -> str:
        self.serial += 1
        return f"{self.serial} {text}"

    def diverge(self) -> None:
        self.diverged = True
        self.steered = False
        self.promised.clear()
        self.trace.start_era()


class ThreadMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.exits = ExitStack()
        self.runner = self.exits.enter_context(asyncio.Runner())
        self.checkpointer = self.exits.enter_context(self.open_checkpointer())
        self.config = ThreadConfig(thread_id="thread")
        self.threads: list[Thread] = []

    def open_checkpointer(self) -> AbstractContextManager[Checkpointer]:
        raise NotImplementedError

    @property
    def thread(self) -> Thread:
        return self.threads[0]

    def teardown(self) -> None:
        self.exits.close()

    @initialize(spec=graph_specs((), 0), origin=st.text(), async_nodes=st.booleans(), data=st.data())
    def build(self, spec: GraphSpec, origin: str, data: st.DataObject, *, async_nodes: bool) -> None:
        names = [node.name for node in spec.nodes]
        stops = st.one_of(st.just("*"), st.lists(st.sampled_from(names), unique=True))
        plain_trace, output = run_plain(spec, origin)
        unfired = find_unfired_joins(spec, plain_trace)
        assert not unfired, f"Expected the target of each join to run after all its sources, not {unfired}."
        trace = Trace()
        interrupt_before = data.draw(stops, label="interrupt_before")
        interrupt_after = data.draw(stops, label="interrupt_after")
        graph = build_graph(spec, Build(interactive=True, async_nodes=async_nodes, trace=trace)).compile(
            checkpointer=self.checkpointer, interrupt_before=interrupt_before, interrupt_after=interrupt_after
        )
        self.threads.append(
            Thread(
                spec,
                graph,
                trace,
                origin,
                canonize(plain_trace, output.get("visits", [])),
                sync_allowed=not (async_nodes and has_async_function(spec)),
                interrupt_before=frozenset(names if interrupt_before == "*" else interrupt_before),
                interrupt_after=frozenset(names if interrupt_after == "*" else interrupt_after),
            )
        )

    def collect_stream(
        self, run_input: State | Command[str] | None, config: ThreadConfig, mode: CallMode, plan: StreamPlan
    ) -> list[StreamPart]:
        graph = self.thread.graph

        def collect_sync() -> list[StreamPart]:
            taken: list[StreamPart] = []
            parts = graph.stream(run_input, config, stream_mode=plan.modes, subgraphs=plan.subgraphs, step_limit=1000)
            with closing(parts):
                for part in parts:
                    if len(taken) == plan.stop_after:
                        break
                    taken.append(part)
            return taken

        match mode:
            case CallMode.asynchronous:
                parts = graph.astream(
                    run_input, config, stream_mode=plan.modes, subgraphs=plan.subgraphs, step_limit=1000
                )
                return self.runner.run(take(parts, plan.stop_after))
            case CallMode.sync:
                return collect_sync()
            case CallMode.sync_in_loop:
                return self.runner.run(call_in_loop(collect_sync))
            case _:
                assert_never(mode)

    def check_stream(self, parts: Sequence[StreamPart], plan: StreamPlan) -> None:
        graph = self.thread.graph
        namespaces = {part.ns for part in parts}
        assert plan.subgraphs or namespaces <= {()}, f"Expected parts of the root graph only, got {namespaces}."
        for namespace in namespaces:
            config = ThreadConfig(thread_id="thread", checkpoint_ns="|".join(namespace))
            found = self.runner.run(graph.aget_state(config))
            assert found.checkpoint_ns == config.get("checkpoint_ns"), f"Expected the state of {namespace}."
        if plan.stop_after is None and StreamMode.values in plan.modes:
            values = [part.values for part in parts if isinstance(part, ValuesPart) and not part.ns]
            newest = self.runner.run(graph.aget_state(self.config)).values
            assert not values or values[-1] == newest, f"Expected the last values part {newest}, got {values[-1]}."

    def deliver(
        self, run_input: State | Command[str] | None, config: ThreadConfig, delivery: Delivery
    ) -> Finished[State] | Interrupted | None:
        graph = self.thread.graph
        mode = delivery.mode if self.thread.sync_allowed else CallMode.asynchronous
        trace = self.thread.trace
        fired = len(trace.fired)
        known = {item.checkpoint_id for item in self.thread.history}
        result: Finished[State] | Interrupted | None = None
        parts: list[StreamPart] | None = None
        raised: list[BaseException] = []
        try:
            if delivery.stream is not None:
                parts = self.collect_stream(run_input, config, mode, delivery.stream)
            else:
                match mode:
                    case CallMode.asynchronous:
                        result = self.runner.run(graph.ainvoke(run_input, config, step_limit=1000))
                    case CallMode.sync:
                        result = graph.invoke(run_input, config, step_limit=1000)
                    case CallMode.sync_in_loop:
                        result = self.runner.run(call_in_loop(lambda: graph.invoke(run_input, config, step_limit=1000)))
                    case _:
                        assert_never(mode)
        except (ConnectionError, ExceptionGroup) as error:
            raised = find_errors(error)
            if not any(isinstance(item, ConnectionError) for item in raised):
                raise
        failed = {path for path, _ in trace.fired[fired:]}
        reported = {item.args[0] for item in raised if isinstance(item, ConnectionError)}
        event(f"failures in one run: {min(len(failed), 2)}")
        assert reported <= failed, f"Expected only the errors of the failures {failed}, got {reported}."
        if delivery.stream is None or delivery.stream.stop_after is None:
            assert reported == failed, f"Expected the errors of all failures {failed}, got {reported}."
        if delivery.stream is None or delivery.stream.stop_after is None:
            tasks = self.runner.run(graph.aget_state(self.config)).tasks
            for path, phase in trace.fired[fired:]:
                name = self.thread.spec.nodes[path[0]].name
                assert phase is Phase.cleanup or any(task.name == name and task.error is not None for task in tasks), (
                    f"Expected the error of the failed node {name!r} in the tasks {tasks}."
                )
        self.check_stops(known)
        if parts is not None and delivery.stream is not None:
            self.check_stream(parts, delivery.stream)
        self.thread.result = result
        return result

    def check_stops(self, known: set[UUID]) -> None:
        thread = self.thread
        thread.history = self.runner.run(collect(thread.graph.aget_state_history(self.config)))
        history = {item.checkpoint_id: item for item in thread.history}
        for reached in thread.history:
            passed = None if reached.parent_id is None else history.get(reached.parent_id)
            if (
                reached.checkpoint_id in known
                or reached.source is not CheckpointSource.loop
                or passed is None
                or passed.checkpoint_id in known
                or passed.source is not CheckpointSource.loop
            ):
                continue
            ran = () if passed.parent_id is None else history[passed.parent_id].next
            assert thread.interrupt_before.isdisjoint(passed.next), (
                f"Expected a stop before {passed.next} for interrupt_before {thread.interrupt_before}."
            )
            assert thread.interrupt_after.isdisjoint(ran), (
                f"Expected a stop after {ran} for interrupt_after {thread.interrupt_after}."
            )

    def is_started(self) -> bool:
        return self.thread.snapshot is not None

    def waiting(self) -> tuple[Interrupt, ...]:
        return () if self.thread.snapshot is None else self.thread.snapshot.interrupts

    def check_questions_still_wait(self, unanswered: Sequence[Interrupt]) -> None:
        now = {
            item.id for item in self.runner.run(self.thread.graph.aget_state(self.config, subgraphs=True)).interrupts
        }
        lost = [item for item in unanswered if item.id not in now]
        assert not lost, f"Expected each question without an answer to wait after the run, lost {lost}."

    @rule(delivery=deliveries(), origin=st.text())
    def send_input(self, delivery: Delivery, origin: str) -> None:
        if self.is_started():
            self.thread.diverge()
            self.deliver(State(origin=origin), self.config, delivery)
        else:
            self.deliver(State(origin=self.thread.origin), self.config, delivery)

    @precondition(is_started)
    @rule(delivery=deliveries())
    def resume(self, delivery: Delivery) -> None:
        unanswered = self.waiting()
        self.deliver(None, self.config, delivery)
        self.check_questions_still_wait(unanswered)

    @precondition(is_started)
    @rule(delivery=deliveries(), picks=st.lists(st.tuples(st.integers(min_value=0), st.text()), min_size=1))
    def answer(self, delivery: Delivery, picks: list[tuple[int, str]]) -> None:
        waiting = self.waiting()
        event(f"answer for waiting interrupts: {bool(waiting)}")
        if not waiting:
            return
        answers: dict[str, str] = {}
        for index, text in picks:
            chosen = waiting[index % len(waiting)]
            if chosen.id not in answers:
                answers[chosen.id] = self.thread.make_text(text)
        self.thread.trace.given.update(
            {read_question(item): answers[item.id] for item in waiting if item.id in answers}
        )
        unanswered = [item for item in waiting if item.id not in answers]
        self.deliver(Command(resume_by_interrupt_id=answers), self.config, delivery)
        self.check_questions_still_wait(unanswered)

    @precondition(is_started)
    @rule(delivery=deliveries(), text=st.text())
    def answer_plainly(self, delivery: Delivery, text: str) -> None:
        thread = self.thread
        answer = thread.make_text(text)
        event(f"plain answer for waiting interrupts: {min(len(self.waiting()), 2)}")
        match self.waiting():
            case []:
                thread.trace.plain = answer
                self.deliver(Command(resume=answer), self.config, delivery)
                thread.trace.plain = None
            case [waiting]:
                thread.trace.given[read_question(waiting)] = answer
                self.deliver(Command(resume=answer), self.config, delivery)
            case _:
                before = self.runner.run(thread.graph.aget_state(self.config))
                with pytest.raises(ValueError):
                    self.deliver(Command(resume=answer), self.config, delivery)
                after = self.runner.run(thread.graph.aget_state(self.config))
                assert after == before, f"Expected a refused answer to leave the state {before}, got {after}."

    @precondition(is_started)
    @rule(delivery=deliveries(), texts=st.lists(st.text()))
    def run_with_an_update(self, delivery: Delivery, texts: list[str]) -> None:
        notes = [self.thread.make_text(text) for text in texts]
        self.thread.diverged = True
        self.thread.promised.extend(notes)
        self.deliver(Command(update=State(visits=notes)), self.config, delivery)

    @precondition(is_started)
    @rule(delivery=deliveries(), data=st.data())
    def run_with_a_goto(self, delivery: Delivery, data: st.DataObject) -> None:
        thread = self.thread
        index = data.draw(st.sampled_from(range(len(thread.spec.nodes))))
        node = thread.spec.nodes[index]
        tag = thread.make_text("") if data.draw(st.booleans()) else None
        snapshot = thread.snapshot
        finished = snapshot is not None and any(
            task.name == node.name and task.result is not None for task in snapshot.tasks
        )
        started = sum(execution.path[0] == index for execution in list(thread.trace.executions.values()))
        thread.diverged = True
        thread.steered = True
        thread.trace.start_era()
        event(f"goto of a command with a send: {tag is not None}")

        target: str | Send = node.name
        if tag is not None:
            match node.work:
                case FunctionSpec():
                    target = Send(node.name, State(origin=tag))
                case ChildSpec():
                    target = Send(node.name, ChildInput(origin=tag, clock=0))
                case _:
                    assert_never(node.work)
        result = self.deliver(Command(goto=target), self.config, delivery)

        executions = list(thread.trace.executions.values())
        now = sum(execution.path[0] == index for execution in executions)
        if result is not None and tag is not None:
            received = [execution.path for execution in executions if execution.origin == tag]
            assert received, f"Expected the node {node.name!r} to run with the payload of the send."
            assert all(path[0] == index for path in received), f"Expected the payload only in {node.name!r}."
        elif result is not None and not finished and not node.defer:
            assert now > started, f"Expected the node {node.name!r} to run after a goto to it."

    @precondition(lambda self: bool(self.thread.history))
    @rule(delivery=deliveries(), data=st.data())
    def run_from_an_older_checkpoint(self, delivery: Delivery, data: st.DataObject) -> None:
        history = self.thread.history
        checkpoint_id = history[data.draw(st.integers(min_value=0)) % len(history)].checkpoint_id
        self.thread.diverge()
        self.deliver(None, ThreadConfig(thread_id="thread", checkpoint_id=checkpoint_id), delivery)

    @rule(data=st.data())
    def arm_faults(self, data: st.DataObject) -> None:
        paths = st.sampled_from(find_function_paths(self.thread.spec))
        self.thread.trace.faults.update(data.draw(st.dictionaries(paths, st.sampled_from(Phase), min_size=1)))

    def next_names(self) -> set[str]:
        thread = self.thread
        if thread.snapshot is None:
            return {thread.spec.nodes[entry].name for entry in thread.spec.entries}
        return set(thread.snapshot.next)

    @precondition(lambda self: len(self.thread.spec.nodes) > 1)
    @rule(data=st.data())
    def arm_faults_of_the_next_step(self, data: st.DataObject) -> None:
        thread = self.thread
        names = self.next_names()
        paths = find_function_paths(thread.spec)
        phases = data.draw(st.lists(st.sampled_from(Phase), min_size=len(paths), max_size=len(paths)))
        event(f"faults for a next step of several nodes: {len(names) > 1}")
        thread.trace.faults.update(
            {path: phase for path, phase in zip(paths, phases, strict=True) if thread.spec.nodes[path[0]].name in names}
        )

    @precondition(is_started)
    @rule(data=st.data(), mode=st.sampled_from(CallMode), texts=st.lists(st.text()))
    def update_state(self, data: st.DataObject, mode: CallMode, texts: list[str]) -> None:
        thread = self.thread
        graph = thread.graph
        snapshot = self.runner.run(graph.aget_state(self.config, subgraphs=True))
        candidates = [snapshot, *find_namespaces(snapshot), *thread.history]
        before = candidates[data.draw(st.integers(min_value=0)) % len(candidates)]
        config = (
            ThreadConfig(thread_id="thread", checkpoint_id=before.checkpoint_id)
            if before in thread.history
            else ThreadConfig(thread_id="thread", checkpoint_ns=before.checkpoint_ns)
        )
        spec = find_spec(thread.spec, before.checkpoint_ns)
        pick = data.draw(st.one_of(st.none(), st.integers(min_value=0)))
        node = None if pick is None else spec.nodes[pick % len(spec.nodes)]
        notes = [thread.make_text(text) for text in texts]
        values = State(visits=notes)
        event(f"state update at an older checkpoint: {before in thread.history[1:]}")
        as_node = None if node is None else node.name
        writer = node
        if node is None and len(spec.nodes) == 1:
            writer = spec.nodes[0]
        elif node is None and before.source is CheckpointSource.loop and before.parent_id is not None:
            parent = self.runner.run(
                graph.aget_state(
                    ThreadConfig(thread_id="thread", checkpoint_ns=before.checkpoint_ns, checkpoint_id=before.parent_id)
                )
            )
            match parent.tasks:
                case [TaskSnapshot(name=name, path=PullPath())]:
                    writer = next((candidate for candidate in spec.nodes if candidate.name == name), None)
                case _:
                    pass
        thread.diverge()
        thread.steered = True
        thread.result = None

        newest_config = ThreadConfig(thread_id="thread", checkpoint_ns=before.checkpoint_ns)
        newest = self.runner.run(graph.aget_state(newest_config))
        updated = True
        try:
            match mode if thread.sync_allowed else CallMode.asynchronous:
                case CallMode.asynchronous:
                    self.runner.run(graph.aupdate_state(config, values, as_node=as_node))
                case CallMode.sync:
                    graph.update_state(config, values, as_node=as_node)
                case CallMode.sync_in_loop:
                    self.runner.run(call_in_loop(lambda: graph.update_state(config, values, as_node=as_node)))
                case _ as unknown:
                    assert_never(unknown)
        except ValueError:
            assert writer is None, f"Expected the update as the node {writer!r} to be accepted."
            updated = False

        after = self.runner.run(graph.aget_state(newest_config))
        if not updated:
            assert after == newest, f"Expected a refused update to leave the state {newest}, got {after}."
            return
        finished = [task.result for task in before.tasks if task.result is not None and "checkpoint_id" not in config]
        event(f"state update over finished tasks: {bool(finished)}")
        expected = [
            *read_visits(before.values),
            *(visit for result in finished for visit in read_visits(result)),
            *notes,
        ]
        assert read_visits(after.values) == expected, f"Expected the visits {expected}, got {after.values}."
        clock = TypeAdapter(int).validate_python(before.values.get("clock", 0)) + sum(
            TypeAdapter(int).validate_python(result.get("clock", 0)) for result in finished
        )
        assert after.values.get("clock", 0) == clock, f"Expected the clock {clock}, got {after.values}."
        if writer is not None:
            targets = {spec.nodes[target].name for target in writer.targets if not spec.nodes[target].defer}
            assert targets <= set(after.next), f"Expected {targets} among the next nodes, got {after.next}."

    @precondition(is_started)
    @rule()
    def finish(self) -> None:
        thread = self.thread
        delivery = Delivery(CallMode.asynchronous, None)
        result = thread.result if isinstance(thread.result, Finished) and not thread.trace.faults else None
        while not isinstance(result, Finished):
            snapshot = self.runner.run(thread.graph.aget_state(self.config))
            progress = (
                snapshot.checkpoint_id,
                snapshot.interrupts,
                len(thread.trace.fired),
                len(thread.trace.consumed),
            )
            answers = {item.id: thread.make_text("") for item in snapshot.interrupts}
            thread.trace.given.update({read_question(item): answers[item.id] for item in snapshot.interrupts})
            result = self.deliver(Command(resume_by_interrupt_id=answers) if answers else None, self.config, delivery)
            if isinstance(result, Finished):
                break
            snapshot = self.runner.run(thread.graph.aget_state(self.config))
            now = (snapshot.checkpoint_id, snapshot.interrupts, len(thread.trace.fired), len(thread.trace.consumed))
            assert now != progress, f"Expected a run with all answers to change the thread, got {result} again."

        visits = result.output.get("visits", [])
        for note in thread.promised:
            assert visits.count(note) == 1, f"Expected the note {note!r} of a command one time in {visits}."
        if not thread.diverged:
            actual = canonize(thread.trace, visits)
            assert actual[0] == thread.expected[0], "Expected the visits of the run without a checkpointer."
            assert actual[1] == thread.expected[1], "Expected the node executions of the run without a checkpointer."
        history = self.runner.run(collect(thread.graph.aget_state_history(self.config)))
        found = [
            self.runner.run(
                thread.graph.aget_state(ThreadConfig(thread_id="thread", checkpoint_id=snapshot.checkpoint_id))
            )
            for snapshot in history
        ]
        assert found == history, (
            "Expected each state read by its checkpoint id to be equal to the state of the history."
        )
        self.record_features()

    def record_features(self) -> None:
        thread = self.thread
        completed = thread.trace.completed()
        paths = Counter((execution.path, execution.era) for execution in completed)
        graphs = dict(walk_graphs(thread.spec, ()))

        def ran(select: Callable[[GraphSpec], Sequence[int]]) -> bool:
            return any(
                (*path, index) in {item.path for item in completed}
                for path, graph in graphs.items()
                for index in select(graph)
            )

        event(f"ran a node two or more times: {any(count > 1 for count in paths.values())}")
        event(
            f"ran a node from a send: {any(visit in thread.trace.tags for item in completed for visit in (*item.observed, item.origin))}"
        )
        event(f"ran a node of a child graph: {any(len(item.path) > 1 for item in completed)}")
        event(f"ran a node two levels down: {any(len(item.path) > 2 for item in completed)}")
        event(
            f"ran the target of a join: {ran(lambda graph: [join.target for join in graph.joins if join.target is not None])}"
        )
        event(
            f"ran a deferred node: {ran(lambda graph: [index for index, node in enumerate(graph.nodes) if node.defer])}"
        )
        event(f"received an answer: {any(item.received for item in completed)}")
        event(f"failed a node: {bool(thread.trace.fired)}")
        event(f"diverged from the plain run: {thread.diverged}")

    @invariant()
    def thread_is_consistent(self) -> None:
        thread = self.thread
        graph = thread.graph
        history = self.runner.run(collect(graph.aget_state_history(self.config)))
        thread.history = history
        if not history:
            return
        snapshot = self.runner.run(graph.aget_state(self.config, subgraphs=True))
        thread.snapshot = snapshot
        newest = self.runner.run(graph.aget_state(self.config))
        assert history[0] == newest, "Expected the newest state of the history to be the state of the thread."

        ids = [item.checkpoint_id for item in history]
        steps = {item.checkpoint_id: item.step for item in history}
        assert ids == sorted(ids, reverse=True), f"Expected the history from the newest checkpoint, got {ids}."
        for item in history:
            if item.parent_id is None:
                expected_step = -1 if item.source is CheckpointSource.input else 0
            else:
                assert item.parent_id in steps, f"Expected the parent {item.parent_id} in the history."
                expected_step = steps[item.parent_id] + (0 if item.source is CheckpointSource.fork else 1)
            assert item.step == expected_step, f"Expected step {expected_step} for {item.source}, got {item.step}."

        match thread.result:
            case Finished(output=output):
                unfired = [] if thread.steered else find_unfired_joins(thread.spec, thread.trace)
                assert not unfired, f"Expected the target of each join to run after all its sources, not {unfired}."
                assert snapshot.next == (), f"Expected no next node after the end, got {snapshot.next}."
                assert snapshot.values == output, f"Expected the output {output}, got {snapshot.values}."
            case Interrupted(interrupts=interrupts, values=values):
                assert snapshot.values == values, f"Expected the values {values}, got {snapshot.values}."
                assert {item.id for item in snapshot.interrupts} == {item.id for item in interrupts}, (
                    f"Expected the interrupts {interrupts}, got {snapshot.interrupts}."
                )
            case None:
                pass
            case _ as unknown:
                assert_never(unknown)

        for child in find_namespaces(snapshot):
            config = ThreadConfig(thread_id="thread", checkpoint_ns=child.checkpoint_ns)
            found = self.runner.run(graph.aget_state(config, subgraphs=True))
            assert found == child, f"Expected the state of the child graph at {child.checkpoint_ns!r}."

        for waiting in snapshot.interrupts:
            assert read_question(waiting) not in thread.trace.consumed, (
                f"Expected no second question for an answer that a node received, got {waiting}."
            )
        completed = thread.trace.completed()
        keys = Counter((item.path, item.era, item.origin, item.clock) for item in completed)
        repeated = [key for key, count in keys.items() if count > 1]
        assert not repeated, f"Expected each task to finish one time, got more for {repeated}."
        for item in completed:
            assert item.received == item.given, (
                f"Expected the answers {item.given} for {item.path}, got {item.received}."
            )


class MemoryThreadMachine(ThreadMachine):
    def open_checkpointer(self) -> AbstractContextManager[Checkpointer]:
        return open_memory()


class SqliteThreadMachine(ThreadMachine):
    def open_checkpointer(self) -> AbstractContextManager[Checkpointer]:
        return open_sqlite()


MemoryThreadMachine.TestCase.settings = settings(deadline=None)
SqliteThreadMachine.TestCase.settings = settings(deadline=None)
TestThreadOnMemory = MemoryThreadMachine.TestCase
TestThreadOnSqlite = SqliteThreadMachine.TestCase


def has_join_as_the_only_way(spec: GraphSpec) -> bool:
    trace, _ = run_plain(spec, "")
    ran = {execution.path for execution in trace.completed()}
    for path, graph in walk_graphs(spec, ()):
        for join in graph.joins:
            target = join.target
            if target is None or (*path, target) not in ran or target in graph.entries:
                continue
            routed = any(
                target in (*node.work.action.forward, *node.work.action.sends, *node.work.action.back)
                for node in graph.nodes
                if isinstance(node.work, FunctionSpec) and isinstance(node.work.action, RouterSpec)
            )
            if not routed and not any(target in node.targets for node in graph.nodes):
                return True
    return False


def runs_a_deferred_node(spec: GraphSpec) -> bool:
    trace, _ = run_plain(spec, "")
    ran = {execution.path for execution in trace.completed()}
    return any(
        node.defer and (*path, index) in ran
        for path, graph in walk_graphs(spec, ())
        for index, node in enumerate(graph.nodes)
    )


def runs_a_node_again(spec: GraphSpec) -> bool:
    trace, _ = run_plain(spec, "")
    return any(count > 1 for count in Counter(execution.path for execution in trace.completed()).values())


def runs_a_send(spec: GraphSpec) -> bool:
    trace, _ = run_plain(spec, "")
    return any(
        execution.origin in trace.tags or set(execution.observed) & set(trace.tags) for execution in trace.completed()
    )


def runs_two_levels_down(spec: GraphSpec) -> bool:
    trace, _ = run_plain(spec, "")
    return any(len(execution.path) > 2 for execution in trace.completed())


def test_generator_reaches_each_way_to_run_a_node() -> None:
    """
    Arrange: the strategy for graphs, and one condition on a run for each way to run a node.
    Act: search the strategy for a graph whose run meets each condition.
    Assert: the search finds a graph for each condition.
    """
    conditions = [
        has_join_as_the_only_way,
        runs_a_deferred_node,
        runs_a_node_again,
        runs_a_send,
        runs_two_levels_down,
    ]

    found = [find(graph_specs((), 0), condition) for condition in conditions]

    assert len(found) == len(conditions), "Expected a graph for each condition."

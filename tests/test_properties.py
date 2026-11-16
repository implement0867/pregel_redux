import asyncio
import threading
from collections import Counter
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Collection,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import AbstractContextManager, ExitStack, aclosing, closing, contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, TypedDict, assert_never
from uuid import UUID

import pytest
from hypothesis import event, example, find, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import create_engine
from typing_extensions import TypeForm

from pregel_redux import (
    Checkpointer,
    Command,
    Depends,
    Endpoint,
    Entrance,
    Finished,
    InMemoryCheckpointer,
    Message,
    PersistentGraph,
    Received,
    Reducer,
    RunAborted,
    Scope,
    Send,
    ShutdownController,
    ShutdownSignal,
    StateGraph,
    StateSnapshot,
    Stopped,
    Waiting,
)
from pregel_redux.checkpoints import CheckpointSource
from pregel_redux.configs import Namespace
from pregel_redux.snapshots import FailedTask, FinishedTask
from pregel_redux.sqlite import SqliteCheckpointer
from pregel_redux.streams import ChildPart, CustomPart, StreamPart, StreamWriter, ValuesPart
from pregel_redux.task_refs import PullRef, PushRef, StartRef
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
class ReceiverSpec:
    pass


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
    action: WriterSpec | ReceiverSpec | RouterSpec
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


class Stop(Enum):
    graceful = "graceful"
    grace_period = "grace_period"
    abort = "abort"


@dataclass
class Execution:
    path: tuple[int, ...]
    era: int
    origin: str | None
    clock: int
    observed: tuple[str, ...]
    received: str | None
    ns: Namespace
    routed: tuple[int, ...] = ()
    completed: bool = False


@dataclass
class Trace:
    executions: dict[str, Execution] = field(default_factory=dict)
    tags: dict[str, tuple[str, int]] = field(default_factory=dict)
    faults: dict[tuple[int, ...], Phase] = field(default_factory=dict)
    delivered: dict[str, tuple[int, ...]] = field(default_factory=dict)
    fired: list[tuple[tuple[int, ...], Phase]] = field(default_factory=list)
    shutdowns: dict[tuple[int, ...], Stop] = field(default_factory=dict)
    controller: ShutdownController = field(default_factory=ShutdownController)
    requested: list[datetime] = field(default_factory=list)
    aborts: int = 0
    era: int = 0
    direct_views: set[tuple[str, ...]] = field(default_factory=set)
    granted: set[tuple[int, ...]] = field(default_factory=set)
    discarded: set[str] = field(default_factory=set)
    command_sends: set[str] = field(default_factory=set)
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

    def request_shutdown(self, path: tuple[int, ...]) -> None:
        with self.lock:
            stop = self.shutdowns.pop(path, None)
            if stop is Stop.graceful:
                self.requested.append(datetime.now(UTC))
            elif stop is not None:
                self.aborts += 1
        match stop:
            case None:
                pass
            case Stop.graceful:
                self.controller.shutdown()
            case Stop.grace_period:
                self.controller.shutdown(grace_period=0)
            case Stop.abort:
                self.controller.abort()
                self.controller.abort()
            case _:
                assert_never(stop)

    def discard_pending_sends(self, kept: Collection[str]) -> None:
        arrived = {run.origin for label, run in list(self.executions.items()) if label in kept}
        self.discarded.update(tag for tag, (sender, _) in self.tags.items() if sender in kept and tag not in arrived)

    def start_era(self) -> None:
        self.era += 1

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


def receives_a_message(node: NodeSpec) -> bool:
    match node.work:
        case FunctionSpec(action=ReceiverSpec()):
            return True
        case FunctionSpec() | ChildSpec():
            return False
        case _:
            assert_never(node.work)


def drop_ways(node: NodeSpec, alone: set[int], receivers: set[int]) -> NodeSpec:
    def keep(indexes: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(index for index in indexes if index not in alone)

    match node.work:
        case FunctionSpec(action=RouterSpec() as router) as function:
            sends = tuple(index for index in keep(router.sends) if index not in receivers)
            action = RouterSpec(keep(router.forward), sends, keep(router.back), router.rounds)
            return replace(node, work=replace(function, action=action), targets=keep(node.targets))
        case FunctionSpec() | ChildSpec():
            return replace(node, targets=keep(node.targets))
        case _:
            assert_never(node.work)


def find_reached_nodes(nodes: Sequence[NodeSpec], entries: Sequence[int]) -> list[int]:
    reached = set(entries)
    for index, node in enumerate(nodes):
        if index in reached:
            reached.update(node.targets)
            match node.work:
                case FunctionSpec(action=RouterSpec() as router):
                    reached.update((*router.forward, *router.sends))
                case FunctionSpec() | ChildSpec():
                    pass
                case _:
                    assert_never(node.work)
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
            st.just(ReceiverSpec()),
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
        node = NodeSpec(node_name, work, draw(st.booleans()), draw(pick_some(later)), draw(st.booleans()))
        nodes.append(replace(node, defer=node.defer and not receives_a_message(node)))
    entries = draw(st.lists(st.sampled_from(range(len(names))), min_size=1, unique=True))
    reached = find_reached_nodes(nodes, entries)
    joins = draw(st.lists(join_specs(len(names), reached))) if len(names) > 2 else []
    targets = sorted({join.target for join in joins if join.target is not None})
    alone = set(draw(pick_some(targets)))
    kept_entries = [entry for entry in entries if entry not in alone] or entries
    receivers = {index for index, node in enumerate(nodes) if receives_a_message(node)}
    return GraphSpec(tuple(drop_ways(node, alone, receivers) for node in nodes), tuple(kept_entries), tuple(joins))


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

    def start(state: State, writer: StreamWriter, received: str | None) -> tuple[Execution, str]:
        execution = Execution(
            path,
            trace.era,
            state.get("origin"),
            state.get("clock", 0),
            tuple(state.get("visits", [])),
            received,
            writer.ns,
        )
        label = trace.begin(execution)
        writer(label)
        trace.request_shutdown(path)
        trace.fail(path, Phase.start)
        return execution, label

    def finish(execution: Execution, label: str) -> object:
        update = State(visits=[label], clock=1)
        result: object = update
        match work.action:
            case WriterSpec() | ReceiverSpec():
                pass
            case RouterSpec() as router:
                rounds = sum(
                    visit in trace.executions and trace.executions[visit].path == path for visit in execution.observed
                )
                back = router.back if rounds < router.rounds else ()
                execution.routed = (*router.forward, *back)
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

    def run_node(state: State, writer: StreamWriter, received: str | None) -> object:
        return finish(*start(state, writer, received))

    async def run_node_async(state: State, writer: StreamWriter, received: str | None) -> object:
        for _ in range(work.pauses):
            await asyncio.sleep(0)
        started = start(state, writer, received)
        for _ in range(work.late_pauses):
            await asyncio.sleep(0)
        return finish(*started)

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

    def act(state: State, writer: StreamWriter) -> object:
        return run_node(state, writer, None)

    async def act_async(state: State, writer: StreamWriter) -> object:
        return await run_node_async(state, writer, None)

    def receive(state: State, writer: StreamWriter, message: Received[str]) -> object:
        return run_node(state, writer, message)

    async def receive_async(state: State, writer: StreamWriter, message: Received[str]) -> object:
        return await run_node_async(state, writer, message)

    is_async = work.is_async and build.async_nodes
    receives = build.interactive and isinstance(work.action, ReceiverSpec)
    if work.resource is None:
        if receives:
            return receive_async if is_async else receive
        return act_async if is_async else act

    provider = open_resource_async if work.resource.is_async and build.async_nodes else open_resource
    scope = work.resource.scope

    def act_with_resource(
        state: State, writer: StreamWriter, resource: Annotated[int, Depends(provider, scope=scope)]
    ) -> object:
        return run_node(state, writer, None)

    async def act_async_with_resource(
        state: State, writer: StreamWriter, resource: Annotated[int, Depends(provider, scope=scope)]
    ) -> object:
        return await run_node_async(state, writer, None)

    def receive_with_resource(
        state: State,
        writer: StreamWriter,
        message: Received[str],
        resource: Annotated[int, Depends(provider, scope=scope)],
    ) -> object:
        return run_node(state, writer, message)

    async def receive_async_with_resource(
        state: State,
        writer: StreamWriter,
        message: Received[str],
        resource: Annotated[int, Depends(provider, scope=scope)],
    ) -> object:
        return await run_node_async(state, writer, message)

    if receives:
        return receive_async_with_resource if is_async else receive_with_resource
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


def find_spec(spec: GraphSpec, checkpoint_ns: Namespace) -> GraphSpec:
    found = spec
    for segment in checkpoint_ns:
        name = segment.node
        match next(node.work for node in found.nodes if node.name == name):
            case ChildSpec(graph=graph):
                found = graph
            case FunctionSpec():
                raise TypeError(f"Expected the namespace {checkpoint_ns!r} to name child graphs, got {name!r}.")
            case _ as unknown:
                assert_never(unknown)
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
    last_runs = {
        join: max((order for order, node in current if node == join.target), default=-1) for join in spec.joins
    }
    return [
        join
        for join in spec.joins
        if join.target is not None and {node for order, node in current if order > last_runs[join]} >= set(join.sources)
    ]


@dataclass(frozen=True)
class Level:
    graph: GraphSpec
    path: tuple[int, ...]
    runs: Mapping[str, Execution]


def find_levels(spec: GraphSpec, trace: Trace) -> list[Level]:
    graphs = dict(walk_graphs(spec, ()))
    groups: dict[tuple[int, Namespace], dict[str, Execution]] = {}
    for label, run in list(trace.executions.items()):
        groups.setdefault((run.era, run.ns), {})[label] = run
    levels: list[Level] = []
    for runs in groups.values():
        path = next(iter(runs.values())).path[:-1]
        levels.append(Level(graphs[path], path, runs))
    return levels


def find_nodes_with_a_way_other_than_a_join(graph: GraphSpec) -> set[int]:
    found = set(graph.entries)
    for node in graph.nodes:
        found.update(node.targets)
        match node.work:
            case FunctionSpec(action=RouterSpec() as router):
                found.update((*router.forward, *router.sends, *router.back))
            case FunctionSpec() | ChildSpec():
                pass
            case _:
                assert_never(node.work)
    return found


def find_sender(trace: Trace, run: Execution) -> str | None:
    if run.origin in trace.tags and run.observed[-1:] == (run.origin,):
        return trace.tags[run.origin][0]
    return None


def is_started_by_a_send(trace: Trace, run: Execution) -> bool:
    return find_sender(trace, run) is not None or (run.origin in trace.command_sends and len(run.path) == 1)


def find_early_deferred(trace: Trace, levels: Sequence[Level]) -> list[tuple[str, str]]:
    early: list[tuple[str, str]] = []
    for level in levels:
        nodes = level.graph.nodes
        late = {
            label
            for label, run in level.runs.items()
            if nodes[run.path[-1]].defer and not is_started_by_a_send(trace, run)
        }
        for late_label in late:
            seen = set(level.runs[late_label].observed)
            for label, run in level.runs.items():
                if not run.completed or nodes[run.path[-1]].defer or label in seen:
                    continue
                sender = find_sender(trace, run)
                if sender is not None:
                    waited = sender not in seen
                elif is_started_by_a_send(trace, run):
                    waited = True
                else:
                    waited = not (late - seen).isdisjoint(run.observed) or (
                        run.observed == level.runs[late_label].observed and run.observed in trace.direct_views
                    )
                if not waited:
                    early.append((late_label, label))
    return early


def find_uncaused(trace: Trace, levels: Sequence[Level]) -> list[str]:
    uncaused: list[str] = []
    for level in levels:
        graph = level.graph
        depth = len(level.path)
        other_ways = find_nodes_with_a_way_other_than_a_join(graph)
        for label, run in level.runs.items():
            index = run.path[-1]
            if (
                is_started_by_a_send(trace, run)
                or run.path in trace.granted
                or run.observed in trace.direct_views
                or (
                    index in graph.entries
                    and not any(
                        seen in level.runs
                        and level.runs[seen].path == run.path
                        and not is_started_by_a_send(trace, level.runs[seen])
                        for seen in run.observed
                    )
                )
            ):
                continue
            known = [
                seen
                for seen in run.observed
                if seen in trace.executions and trace.executions[seen].path[:depth] == level.path
            ]
            earlier = [
                seen
                for seen in known
                if trace.executions[seen].path == run.path and not is_started_by_a_send(trace, trace.executions[seen])
            ]
            old = set(trace.executions[earlier[-1]].observed) if earlier else set[str]()
            fresh = [trace.executions[seen] for seen in known if seen not in old]
            finished = {trace.executions[seen].path[depth] for seen in known}
            finished_again = {item.path[depth] for item in fresh}
            routed = {target for item in fresh if item.path[:-1] == level.path for target in item.routed}
            joins = [join for join in graph.joins if join.target == index]
            only_join = index not in other_ways and len(joins) == 1
            by_join = any(
                finished_again >= set(join.sources)
                if only_join
                else finished >= set(join.sources) and not finished_again.isdisjoint(join.sources)
                for join in joins
            )
            by_edge = any(index in graph.nodes[source].targets for source in finished_again)
            if not (by_edge or by_join or index in routed):
                uncaused.append(label)
    return uncaused


def find_wrong_views(trace: Trace, levels: Sequence[Level]) -> list[tuple[str, str]]:
    wrong: list[tuple[str, str]] = []
    for label, run in list(trace.executions.items()):
        wrong.extend((label, seen) for seen, count in Counter(run.observed).items() if count > 1)
        wrong.extend(
            (label, seen) for seen in run.observed if seen in trace.executions and not trace.executions[seen].completed
        )
    for level in levels:
        pulls = {label: run for label, run in level.runs.items() if not is_started_by_a_send(trace, run)}
        for label, run in pulls.items():
            for seen in run.observed:
                if seen not in pulls:
                    continue
                earlier = pulls[seen].observed
                if run.observed[: len(earlier)] != earlier:
                    wrong.append((label, seen))
                wrong.extend(
                    (label, other)
                    for other, peer in pulls.items()
                    if peer.completed and peer.observed == earlier and other not in run.observed
                )
    return wrong


def find_wrong_sends(spec: GraphSpec, trace: Trace, visits: Sequence[str] | None) -> list[tuple[str, str]]:
    graphs = dict(walk_graphs(spec, ()))
    runs = list(trace.executions.items())
    wrong: list[tuple[str, str]] = []
    for tag, (sender_label, position) in list(trace.tags.items()):
        sender = trace.executions[sender_label]
        graph = graphs[sender.path[:-1]]
        match graph.nodes[sender.path[-1]].work:
            case FunctionSpec(action=RouterSpec(sends=sends)):
                target = (*sender.path[:-1], sends[position])
            case _:
                raise TypeError(f"Expected the sender of {tag!r} to be a router.")
        started = [(label, run) for label, run in runs if run.origin == tag]
        arrived = (
            None
            if visits is None or sender_label not in visits or tag in trace.discarded
            else [label for label, _ in started if label in visits]
        )
        match graph.nodes[target[-1]].work:
            case FunctionSpec():
                expected = (target, sender.ns, sender.clock, (*sender.observed, sender_label, tag))
                wrong.extend(
                    (tag, label) for label, run in started if (run.path, run.ns, run.clock, run.observed) != expected
                )
                if arrived is not None and len(arrived) != 1:
                    wrong.append((tag, sender_label))
            case ChildSpec():
                wrong.extend(
                    (tag, label)
                    for label, run in started
                    if run.path[: len(target)] != target or len(run.path) == len(target) or tag in run.observed
                )
                if arrived is not None and not arrived:
                    wrong.append((tag, sender_label))
            case _ as unknown:
                assert_never(unknown)
    return wrong


def check_order_rules(spec: GraphSpec, trace: Trace, visits: Sequence[str] | None) -> None:
    levels = find_levels(spec, trace)
    broken = {
        "a deferred node waits until no other node can run": find_early_deferred(trace, levels),
        "an edge, a goto, a send or a full join comes before each node execution": find_uncaused(trace, levels),
        "a node sees each write of the earlier steps one time": find_wrong_views(trace, levels),
        "a send starts one execution of its node with its payload": find_wrong_sends(spec, trace, visits),
    }
    failures = {rule: found for rule, found in broken.items() if found}
    assert not failures, f"Expected the trace to follow each order rule, got {failures} in {trace.executions}."


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
    stop_after: int | None


@dataclass(frozen=True)
class Delivery:
    mode: CallMode
    stream: StreamPlan | None


def deliveries() -> st.SearchStrategy[Delivery]:
    plan = st.builds(
        StreamPlan,
        stop_after=st.one_of(st.none(), st.integers(min_value=0)),
    )
    return st.builds(Delivery, mode=st.sampled_from(CallMode), stream=st.one_of(st.none(), plan))


async def take(parts: AsyncGenerator[StreamPart | ChildPart], count: int | None) -> list[StreamPart | ChildPart]:
    taken: list[StreamPart | ChildPart] = []
    async with aclosing(parts):
        async for part in parts:
            if len(taken) == count:
                break
            taken.append(part)
    return taken


async def call_in_loop[ResultT](function: Callable[[], ResultT]) -> ResultT:
    return function()


def find_path(spec: GraphSpec, entrance: Entrance) -> tuple[int, ...]:
    found = spec
    path: list[int] = []
    for name in (*(segment.node for segment in entrance.checkpoint_ns), entrance.node):
        index = next(index for index, node in enumerate(found.nodes) if node.name == name)
        path.append(index)
        match found.nodes[index].work:
            case ChildSpec(graph=graph):
                found = graph
            case FunctionSpec():
                pass
            case _ as unknown:
                assert_never(unknown)
    return tuple(path)


def find_level_path(spec: GraphSpec, checkpoint_ns: Namespace) -> tuple[int, ...]:
    found = spec
    path: list[int] = []
    for segment in checkpoint_ns:
        index = next(index for index, node in enumerate(found.nodes) if node.name == segment.node)
        path.append(index)
        found = find_spec(found, (segment,))
    return tuple(path)


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
            case BaseException():
                pass
            case _ as unknown:
                assert_never(unknown)
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
    commanded: bool = False
    result: Finished[State] | Waiting | Stopped | None = None
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
        self.thread_id = "thread"
        self.threads: list[Thread] = []

    def open_checkpointer(self) -> AbstractContextManager[Checkpointer]:
        raise NotImplementedError

    @property
    def thread(self) -> Thread:
        return self.threads[0]

    def teardown(self) -> None:
        self.exits.close()

    @initialize(
        spec=graph_specs((), 0), origin=st.text(), thread_id=st.text(), async_nodes=st.booleans(), data=st.data()
    )
    def build(self, spec: GraphSpec, origin: str, thread_id: str, data: st.DataObject, *, async_nodes: bool) -> None:
        self.thread_id = thread_id
        names = [node.name for node in spec.nodes]
        stops = st.one_of(st.just("*"), st.lists(st.sampled_from(names), unique=True))
        plain_trace, output = run_plain(spec, origin)
        unfired = find_unfired_joins(spec, plain_trace)
        assert not unfired, f"Expected the target of each join to run after all its sources, not {unfired}."
        check_order_rules(spec, plain_trace, output.get("visits", []))
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
        self,
        run_input: State | Command[str] | Message | None,
        fork_from: UUID | None,
        mode: CallMode,
        plan: StreamPlan,
        signal: ShutdownSignal,
    ) -> list[StreamPart | ChildPart]:
        graph = self.thread.graph

        def collect_sync() -> list[StreamPart | ChildPart]:
            taken: list[StreamPart | ChildPart] = []
            parts = graph.stream(run_input, self.thread_id, fork_from=fork_from, step_limit=1000, signal=signal)
            with closing(parts):
                for part in parts:
                    if len(taken) == plan.stop_after:
                        break
                    taken.append(part)
            return taken

        match mode:
            case CallMode.asynchronous:
                parts = graph.astream(run_input, self.thread_id, fork_from=fork_from, step_limit=1000, signal=signal)
                return self.runner.run(take(parts, plan.stop_after))
            case CallMode.sync:
                return collect_sync()
            case CallMode.sync_in_loop:
                return self.runner.run(call_in_loop(collect_sync))
            case _:
                assert_never(mode)

    def check_stream(self, parts: Sequence[StreamPart | ChildPart], plan: StreamPlan, begun: int) -> None:
        graph = self.thread.graph
        written: list[object] = []
        namespaces = {part.ns for part in parts if isinstance(part, ChildPart)}
        for namespace in namespaces:
            found = self.runner.run(graph.aget_state(self.thread_id, checkpoint_ns=namespace))
            assert found.checkpoint_ns == namespace, f"Expected the state of {namespace}."
        levels: dict[Namespace, list[object]] = {}
        for part in parts:
            match part:
                case ChildPart(part=CustomPart(data=data)) | CustomPart(data=data):
                    written.append(data)
                case ChildPart(ns=ns, part=ValuesPart(values=child_values)):
                    levels.setdefault(ns, []).append(child_values)
                case ValuesPart(values=root_values):
                    levels.setdefault((), []).append(root_values)
                case _:
                    pass
        labels = [f"#{index}" for index in range(begun, len(self.thread.trace.executions))]
        if plan.stop_after is None:
            assert sorted(written, key=repr) == sorted(labels), (
                f"Expected one custom part for each node execution {labels}, got {written}."
            )
        for namespace, sent in levels.items():
            saved = [
                item.values
                for item in self.runner.run(collect(graph.aget_state_history(self.thread_id, checkpoint_ns=namespace)))
            ]
            unknown = [values for values in sent if values not in saved]
            assert not unknown, f"Expected each values part of {namespace!r} to be a saved state, got {unknown}."
        if plan.stop_after is None:
            values = [part.values for part in parts if isinstance(part, ValuesPart)]
            newest = self.runner.run(graph.aget_state(self.thread_id)).values
            assert not values or values[-1] == newest, f"Expected the last values part {newest}, got {values[-1]}."

    def deliver(
        self, run_input: State | Command[str] | Message | None, fork_from: UUID | None, delivery: Delivery
    ) -> Finished[State] | Waiting | Stopped | None:
        graph = self.thread.graph
        mode = delivery.mode if self.thread.sync_allowed else CallMode.asynchronous
        trace = self.thread.trace
        fired = len(trace.fired)
        requested = len(trace.requested)
        begun = len(trace.executions)
        aborts = trace.aborts
        aborted = False
        trace.controller = ShutdownController()
        signal = trace.controller.signal
        known = {item.checkpoint_id for item in self.thread.history}
        stored = self.runner.run(collect(graph.aget_state_history(self.thread_id)))
        assert stored == self.thread.history, "Expected no change of the stored checkpoints between two runs."
        result: Finished[State] | Waiting | Stopped | None = None
        parts: list[StreamPart | ChildPart] | None = None
        raised: list[BaseException] = []
        try:
            if delivery.stream is not None:
                parts = self.collect_stream(run_input, fork_from, mode, delivery.stream, signal)
            else:
                match mode:
                    case CallMode.asynchronous:
                        result = self.runner.run(
                            graph.ainvoke(
                                run_input, self.thread_id, fork_from=fork_from, step_limit=1000, signal=signal
                            )
                        )
                    case CallMode.sync:
                        result = graph.invoke(
                            run_input, self.thread_id, fork_from=fork_from, step_limit=1000, signal=signal
                        )
                    case CallMode.sync_in_loop:
                        result = self.runner.run(
                            call_in_loop(
                                lambda: graph.invoke(
                                    run_input, self.thread_id, fork_from=fork_from, step_limit=1000, signal=signal
                                )
                            )
                        )
                    case _:
                        assert_never(mode)
        except RunAborted:
            aborted = True
        except (ConnectionError, ExceptionGroup) as error:
            raised = find_errors(error)
            if not any(isinstance(item, ConnectionError) for item in raised):
                raise
        event(f"abort of a run: {aborted}")
        assert not aborted or trace.aborts > aborts, "Expected RunAborted only after an abort request."
        failed = {path for path, _ in trace.fired[fired:]}
        reported = {item.args[0] for item in raised if isinstance(item, ConnectionError)}
        event(f"failures in one run: {min(len(failed), 2)}")
        assert reported <= failed, f"Expected only the errors of the failures {failed}, got {reported}."
        complete = trace.aborts == aborts and (delivery.stream is None or delivery.stream.stop_after is None)
        if complete:
            assert reported == failed, f"Expected the errors of all failures {failed}, got {reported}."
        if complete:
            tasks = self.runner.run(graph.aget_state(self.thread_id)).tasks
            for path, phase in trace.fired[fired:]:
                name = self.thread.spec.nodes[path[0]].name
                assert phase is Phase.cleanup or any(
                    task.name == name and isinstance(task.status, FailedTask) for task in tasks
                ), f"Expected the error of the failed node {name!r} in the tasks {tasks}."
        self.check_stops(known)
        self.check_shutdown(trace.requested[requested:])
        if parts is not None and delivery.stream is not None:
            self.check_stream(parts, delivery.stream, begun)
        self.thread.result = result
        return result

    def check_shutdown(self, requests: Sequence[datetime]) -> None:
        event(f"graceful shutdown in a run: {bool(requests)}")
        if not requests:
            return
        graph = self.thread.graph
        snapshot = self.runner.run(graph.aget_state(self.thread_id, subgraphs=True))
        for namespace in [(), *(child.checkpoint_ns for child in find_namespaces(snapshot))]:
            later = [
                item.step
                for item in self.runner.run(collect(graph.aget_state_history(self.thread_id, checkpoint_ns=namespace)))
                if item.source is CheckpointSource.loop and item.created_at > requests[0]
            ]
            assert len(later) <= 1, (
                f"Expected the graph {namespace!r} to stop after its current step at a shutdown, got the steps {later}."
            )

    def check_stops(self, known: set[UUID]) -> None:
        thread = self.thread
        thread.history = self.runner.run(collect(thread.graph.aget_state_history(self.thread_id)))
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

    def accepts(self) -> Mapping[Entrance, TypeForm[object]]:
        return {} if self.thread.snapshot is None else self.thread.snapshot.accepts

    def check_entrances_still_wait(self, unanswered: Sequence[Entrance]) -> None:
        graph = self.thread.graph
        for entrance in unanswered:
            level = self.runner.run(graph.aget_state(self.thread_id, checkpoint_ns=entrance.checkpoint_ns))
            assert level.next or entrance in level.accepts, (
                f"Expected the entrance {entrance} without a message to stay open in its graph at rest, got {level}."
            )

    def send_message(
        self, entrance: Entrance, fork_from: UUID | None, delivery: Delivery
    ) -> Finished[State] | Waiting | Stopped | None:
        thread = self.thread
        text = thread.make_text("")
        thread.trace.delivered[text] = find_path(thread.spec, entrance)
        thread.diverged = True
        result = self.deliver(Message(text, entrance), fork_from, delivery)
        match result:
            case Finished() | Waiting():
                received = [item for item in thread.trace.completed() if item.received == text]
                assert received, f"Expected the node at {entrance} to receive the message {text!r} in its run."
            case Stopped() | None:
                pass
            case _:
                assert_never(result)
        return result

    @rule(delivery=deliveries(), origin=st.text())
    def send_input(self, delivery: Delivery, origin: str) -> None:
        snapshot = self.thread.snapshot
        if snapshot is not None:
            self.thread.trace.discard_pending_sends(read_visits(snapshot.values))
            self.thread.diverge()
            self.deliver(State(origin=origin), None, delivery)
        else:
            self.deliver(State(origin=self.thread.origin), None, delivery)

    @precondition(is_started)
    @rule(delivery=deliveries())
    def resume(self, delivery: Delivery) -> None:
        unanswered = list(self.accepts())
        self.deliver(None, None, delivery)
        self.check_entrances_still_wait(unanswered)

    @precondition(is_started)
    @rule(delivery=deliveries(), pick=st.integers(min_value=0), from_newest=st.booleans())
    def answer(self, delivery: Delivery, pick: int, *, from_newest: bool) -> None:
        thread = self.thread
        waiting = list(self.accepts())
        event(f"message for an open entrance: {bool(waiting)}")
        if not waiting:
            return
        chosen = waiting[pick % len(waiting)]
        unanswered = [entrance for entrance in waiting if entrance != chosen]
        newest = thread.history[0].checkpoint_id if from_newest else None
        event(f"message with the newest checkpoint id: {from_newest}")
        if from_newest and chosen.checkpoint_ns:
            before = self.runner.run(thread.graph.aget_state(self.thread_id))
            with pytest.raises(ValueError):
                self.deliver(Message(thread.make_text(""), chosen), newest, delivery)
            after = self.runner.run(thread.graph.aget_state(self.thread_id))
            assert after == before, f"Expected a refused message to leave the state {before}, got {after}."
            return
        self.send_message(chosen, newest, delivery)
        if not from_newest:
            self.check_entrances_still_wait(unanswered)

    @precondition(is_started)
    @rule(delivery=deliveries())
    def answer_without_an_entrance(self, delivery: Delivery) -> None:
        thread = self.thread
        waiting = list(self.accepts())
        event(f"message without an entrance for open entrances: {min(len(waiting), 2)}")
        match waiting:
            case [entrance]:
                text = thread.make_text("")
                thread.trace.delivered[text] = find_path(thread.spec, entrance)
                thread.diverged = True
                self.deliver(Message(text), None, delivery)
            case _:
                before = self.runner.run(thread.graph.aget_state(self.thread_id))
                with pytest.raises(ValueError):
                    self.deliver(Message(thread.make_text("")), None, delivery)
                after = self.runner.run(thread.graph.aget_state(self.thread_id))
                assert after == before, f"Expected a refused message to leave the state {before}, got {after}."

    @precondition(is_started)
    @rule(delivery=deliveries(), texts=st.lists(st.text()))
    def run_with_an_update(self, delivery: Delivery, texts: list[str]) -> None:
        notes = [self.thread.make_text(text) for text in texts]
        self.thread.diverged = True
        self.thread.commanded = True
        self.thread.promised.extend(notes)
        self.deliver(Command(update=State(visits=notes)), None, delivery)

    @precondition(is_started)
    @rule(delivery=deliveries(), data=st.data())
    def run_with_a_goto(self, delivery: Delivery, data: st.DataObject) -> None:
        thread = self.thread
        index = data.draw(st.sampled_from(range(len(thread.spec.nodes))))
        node = thread.spec.nodes[index]
        tag = thread.make_text("") if data.draw(st.booleans()) else None
        snapshot = thread.snapshot
        finished = snapshot is not None and any(
            task.name == node.name and isinstance(task.status, FinishedTask) for task in snapshot.tasks
        )
        started = sum(execution.path[0] == index for execution in list(thread.trace.executions.values()))
        thread.diverged = True
        thread.steered = True
        thread.commanded = True
        thread.trace.start_era()
        thread.trace.granted.add((index,))
        if snapshot is not None:
            thread.trace.direct_views.add(tuple(read_visits(snapshot.values)))
        event(f"goto of a command with a send: {tag is not None}")

        target: str | Send = node.name
        if tag is not None and not receives_a_message(node):
            thread.trace.command_sends.add(tag)
            match node.work:
                case FunctionSpec():
                    target = Send(node.name, State(origin=tag))
                case ChildSpec():
                    target = Send(node.name, ChildInput(origin=tag, clock=0))
                case _:
                    assert_never(node.work)
        requested = len(thread.trace.requested)
        unanswered = list(self.accepts())
        known = {item.checkpoint_id for item in thread.history}
        result = self.deliver(Command(goto=target), None, delivery)
        self.check_entrances_still_wait(unanswered)
        if len(thread.trace.requested) > requested:
            return

        executions = list(thread.trace.executions.values())
        now = sum(execution.path[0] == index for execution in executions)
        if isinstance(result, Finished | Stopped) and tag is not None and not receives_a_message(node):
            received = [execution.path for execution in executions if execution.origin == tag]
            assert received, f"Expected the node {node.name!r} to run with the payload of the send."
            assert all(path[0] == index for path in received), f"Expected the payload only in {node.name!r}."
        elif (
            isinstance(result, Finished | Stopped) and not finished and not node.defer and not receives_a_message(node)
        ):
            ran: dict[UUID | None, tuple[str, ...]] = {item.checkpoint_id: item.next for item in thread.history}
            committed = any(
                item.checkpoint_id not in known
                and item.source is CheckpointSource.loop
                and node.name in ran.get(item.parent_id, ())
                for item in thread.history
            )
            assert now > started or committed, (
                f"Expected the node {node.name!r} to run, or its step to finish with saved writes, after a goto to it."
            )

    @precondition(lambda self: bool(self.thread.history))
    @rule(delivery=deliveries(), data=st.data())
    def run_from_an_older_checkpoint(self, delivery: Delivery, data: st.DataObject) -> None:
        history = self.thread.history
        checkpoint_id = history[data.draw(st.integers(min_value=0)) % len(history)].checkpoint_id
        self.thread.diverge()
        self.deliver(None, checkpoint_id, delivery)

    @precondition(lambda self: bool(self.accepts()))
    @rule(pick=st.integers(min_value=0))
    def answer_with_a_wrong_type(self, pick: int) -> None:
        thread = self.thread
        waiting = list(self.accepts())
        chosen = waiting[pick % len(waiting)]
        before = self.runner.run(thread.graph.aget_state(self.thread_id, subgraphs=True))
        with pytest.raises(ValidationError):
            self.runner.run(thread.graph.ainvoke(Message(0, chosen), self.thread_id, step_limit=1000))
        after = self.runner.run(thread.graph.aget_state(self.thread_id, subgraphs=True))
        assert after == before, f"Expected a refused message to leave the state {before}, got {after}."

    @precondition(is_started)
    @rule()
    def delete_the_thread(self) -> None:
        thread = self.thread
        self.checkpointer.delete_thread(self.thread_id)
        history = self.runner.run(collect(thread.graph.aget_state_history(self.thread_id)))
        assert history == [], f"Expected no state of a deleted thread, got {history}."
        with pytest.raises(LookupError):
            self.runner.run(thread.graph.aget_state(self.thread_id))
        thread.diverge()
        thread.result = None
        thread.snapshot = None
        thread.history = []

    @precondition(lambda self: bool(self.thread.history))
    @rule(limit=st.integers(min_value=0), start=st.integers(min_value=0))
    def read_a_part_of_the_history(self, limit: int, start: int) -> None:
        graph = self.thread.graph
        history = self.thread.history
        limited = self.runner.run(collect(graph.aget_state_history(self.thread_id, limit=limit)))
        assert limited == history[:limit], f"Expected the newest {limit} states of the history."
        position = start % len(history)
        older = self.runner.run(
            collect(graph.aget_state_history(self.thread_id, before=history[position].checkpoint_id))
        )
        assert older == history[position + 1 :], f"Expected the states before the state at {position}."
        with pytest.raises(ValueError):
            self.runner.run(collect(graph.aget_state_history(self.thread_id, limit=-1 - limit)))

    @rule(delivery=deliveries(), checkpoint_id=st.uuids(), update=st.booleans())
    def use_an_unknown_checkpoint(self, delivery: Delivery, checkpoint_id: UUID, *, update: bool) -> None:
        graph = self.thread.graph
        before = self.runner.run(collect(graph.aget_state_history(self.thread_id)))
        with pytest.raises(LookupError):
            if update:
                self.runner.run(
                    graph.aupdate_state(
                        self.thread_id, State(), as_node=self.thread.spec.nodes[0].name, checkpoint_id=checkpoint_id
                    )
                )
            else:
                self.deliver(None, checkpoint_id, delivery)
        after = self.runner.run(collect(graph.aget_state_history(self.thread_id)))
        assert after == before, "Expected a refused checkpoint id to leave the thread as it was."

    @rule(data=st.data())
    def arm_shutdowns(self, data: st.DataObject) -> None:
        paths = st.sampled_from(find_function_paths(self.thread.spec))
        self.thread.trace.shutdowns.update(data.draw(st.dictionaries(paths, st.sampled_from(Stop), min_size=1)))

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
        snapshot = self.runner.run(graph.aget_state(self.thread_id, subgraphs=True))
        candidates = [snapshot, *find_namespaces(snapshot), *thread.history]
        before = candidates[data.draw(st.integers(min_value=0)) % len(candidates)]
        first_steps = [item for item in thread.history if item.step == 1 and item.source is CheckpointSource.loop]
        if data.draw(st.booleans(), label="after the first step") and first_steps:
            before = first_steps[-1]
        older = before in thread.history
        checkpoint_id = before.checkpoint_id if older else None
        checkpoint_ns = () if older else before.checkpoint_ns
        spec = find_spec(thread.spec, before.checkpoint_ns)
        pick = data.draw(st.one_of(st.none(), st.integers(min_value=0)))
        node = None if pick is None else spec.nodes[pick % len(spec.nodes)]
        notes = [thread.make_text(text) for text in texts]
        values = State(visits=notes)
        event(f"state update at an older checkpoint: {before in thread.history[1:]}")
        as_node = None if node is None else node.name
        writer = node
        unclear = False
        if node is None and len(spec.nodes) == 1:
            writer = spec.nodes[0]
        elif node is None and before.source is CheckpointSource.loop and before.parent_id is not None:
            parent = self.runner.run(
                graph.aget_state(self.thread_id, checkpoint_ns=before.checkpoint_ns, checkpoint_id=before.parent_id)
            )
            origin = (
                None
                if parent.parent_id is None
                else self.runner.run(
                    graph.aget_state(self.thread_id, checkpoint_ns=before.checkpoint_ns, checkpoint_id=parent.parent_id)
                )
            )
            unclear = (
                origin is not None
                and origin.source is CheckpointSource.input
                and all(isinstance(task.path, PullRef) for task in parent.tasks)
                and len({task.name for task in parent.tasks}) > 1
            )
            if len(parent.tasks) == 1:
                task = parent.tasks[0]
                match task.path:
                    case PullRef():
                        writer = next((candidate for candidate in spec.nodes if candidate.name == task.name), None)
                    case PushRef() | StartRef():
                        pass
                    case _ as unknown:
                        assert_never(unknown)
        guessed_past_a_wait = node is None and len(spec.nodes) > 1 and any(map(receives_a_message, spec.nodes))
        event(f"state update without a node in a graph with a node that waits: {guessed_past_a_wait}")
        thread.diverge()
        thread.steered = True
        thread.commanded = True
        thread.result = None

        newest = self.runner.run(graph.aget_state(self.thread_id, checkpoint_ns=before.checkpoint_ns))
        updated = True
        try:
            match mode if thread.sync_allowed else CallMode.asynchronous:
                case CallMode.asynchronous:
                    self.runner.run(
                        graph.aupdate_state(
                            self.thread_id,
                            values,
                            as_node=as_node,
                            checkpoint_ns=checkpoint_ns,
                            checkpoint_id=checkpoint_id,
                        )
                    )
                case CallMode.sync:
                    graph.update_state(
                        self.thread_id,
                        values,
                        as_node=as_node,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                    )
                case CallMode.sync_in_loop:
                    self.runner.run(
                        call_in_loop(
                            lambda: graph.update_state(
                                self.thread_id,
                                values,
                                as_node=as_node,
                                checkpoint_ns=checkpoint_ns,
                                checkpoint_id=checkpoint_id,
                            )
                        )
                    )
                case _ as unknown:
                    assert_never(unknown)
        except ValueError:
            assert writer is None or guessed_past_a_wait, f"Expected the update as the node {writer!r} to be accepted."
            updated = False

        after = self.runner.run(graph.aget_state(self.thread_id, checkpoint_ns=before.checkpoint_ns))
        event(f"state update without a node after a step of several nodes: {unclear}")
        assert not (updated and unclear) or guessed_past_a_wait, (
            "Expected an update without a node after a step of several nodes to raise."
        )
        if not updated:
            assert after == newest, f"Expected a refused update to leave the state {newest}, got {after}."
            return
        thread.trace.discard_pending_sends(read_visits(before.values))
        thread.trace.direct_views.add(tuple(read_visits(after.values)))
        level = find_level_path(thread.spec, before.checkpoint_ns)
        source = None if node is None else spec.nodes.index(node)
        thread.trace.granted.update(
            (*level, index)
            for index in range(len(spec.nodes))
            if source is None
            or index in spec.nodes[source].targets
            or any(join.target == index and source in join.sources for join in spec.joins)
        )
        finished = [
            task.status.result
            for task in before.tasks
            if isinstance(task.status, FinishedTask) and checkpoint_id is None
        ]
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
        if writer is not None and not guessed_past_a_wait:
            targets = {
                spec.nodes[target].name
                for target in writer.targets
                if not spec.nodes[target].defer and not receives_a_message(spec.nodes[target])
            }
            assert targets <= set(after.next), f"Expected {targets} among the next nodes, got {after.next}."

    @precondition(is_started)
    @rule()
    def finish(self) -> None:
        thread = self.thread
        delivery = Delivery(CallMode.asynchronous, None)
        result = None if thread.trace.faults else thread.result
        while True:
            match result:
                case Finished():
                    break
                case Waiting() | Stopped() | None:
                    pass
                case _:
                    assert_never(result)
            snapshot = self.runner.run(thread.graph.aget_state(self.thread_id))
            trace = thread.trace
            progress = (
                snapshot.checkpoint_id,
                snapshot.accepts,
                len(trace.fired),
                len(trace.completed()),
                len(trace.shutdowns),
            )
            entrances = list(snapshot.accepts)
            if entrances:
                result = self.send_message(entrances[0], None, delivery)
            else:
                result = self.deliver(None, None, delivery)
            match result:
                case Finished():
                    pass
                case Waiting() | Stopped() | None:
                    snapshot = self.runner.run(thread.graph.aget_state(self.thread_id))
                    now = (
                        snapshot.checkpoint_id,
                        snapshot.accepts,
                        len(trace.fired),
                        len(trace.completed()),
                        len(trace.shutdowns),
                    )
                    assert now != progress, f"Expected a run to change the thread, got {result} again."
                case _:
                    assert_never(result)

        visits = result.output.get("visits", [])
        for note in thread.promised:
            assert visits.count(note) == 1, f"Expected the note {note!r} of a command one time in {visits}."
        if not thread.diverged:
            actual = canonize(thread.trace, visits)
            assert actual[0] == thread.expected[0], "Expected the visits of the run without a checkpointer."
            assert actual[1] == thread.expected[1], "Expected the node executions of the run without a checkpointer."
        history = self.runner.run(collect(thread.graph.aget_state_history(self.thread_id)))
        found = [
            self.runner.run(thread.graph.aget_state(self.thread_id, checkpoint_id=snapshot.checkpoint_id))
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
        event(f"received a message: {any(item.received is not None for item in completed)}")
        event(f"failed a node: {bool(thread.trace.fired)}")
        event(f"diverged from the plain run: {thread.diverged}")

    @invariant()
    def thread_is_consistent(self) -> None:
        thread = self.thread
        graph = thread.graph
        match thread.result:
            case Finished(output=finished):
                check_order_rules(thread.spec, thread.trace, finished.get("visits", []))
            case Waiting() | Stopped() | None:
                check_order_rules(thread.spec, thread.trace, None)
            case _ as unknown:
                assert_never(unknown)
        history = self.runner.run(collect(graph.aget_state_history(self.thread_id)))
        thread.history = history
        if not history:
            return
        snapshot = self.runner.run(graph.aget_state(self.thread_id, subgraphs=True))
        thread.snapshot = snapshot
        newest = self.runner.run(graph.aget_state(self.thread_id))
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
                assert snapshot.accepts == {}, f"Expected no open entrance after the end, got {snapshot.accepts}."
                assert snapshot.values == output, f"Expected the output {output}, got {snapshot.values}."
            case Waiting(values=values, accepts=accepts):
                assert snapshot.values == values, f"Expected the values {values}, got {snapshot.values}."
                assert snapshot.accepts == accepts, f"Expected the entrances {accepts}, got {snapshot.accepts}."
            case Stopped(values=values):
                assert snapshot.values == values, f"Expected the values {values}, got {snapshot.values}."
            case None:
                pass
            case _ as unknown:
                assert_never(unknown)

        for child in find_namespaces(snapshot):
            found = self.runner.run(graph.aget_state(self.thread_id, checkpoint_ns=child.checkpoint_ns, subgraphs=True))
            assert found == child, f"Expected the state of the child graph at {child.checkpoint_ns!r}."

        for level in [snapshot, *find_namespaces(snapshot)]:
            deferred = {node.name for node in find_spec(thread.spec, level.checkpoint_ns).nodes if node.defer}
            pulled = {task.name for task in level.tasks if isinstance(task.path, PullRef)}
            assert thread.commanded or not deferred & pulled or deferred >= pulled, (
                f"Expected a deferred node to wait until no other node runs, got the tasks {level.tasks}."
            )

        completed = thread.trace.completed()
        keys = Counter((item.path, item.era, item.origin, item.clock) for item in completed)
        repeated = [key for key, count in keys.items() if count > 1]
        assert not repeated, f"Expected each task to finish one time, got more for {repeated}."
        messages = Counter((item.era, item.received) for item in completed if item.received is not None)
        repeated_messages = [key for key, count in messages.items() if count > 1]
        assert not repeated_messages, f"Expected each message to reach one node execution, got {repeated_messages}."
        for item in completed:
            if item.received is not None:
                expected_path = thread.trace.delivered.get(item.received)
                assert item.path == expected_path, (
                    f"Expected the message {item.received!r} at {expected_path}, got it at {item.path}."
                )
        for level in [snapshot, *find_namespaces(snapshot)]:
            receivers = {
                node.name for node in find_spec(thread.spec, level.checkpoint_ns).nodes if receives_a_message(node)
            }
            offered = {entrance.node for entrance in level.accepts if entrance.checkpoint_ns == level.checkpoint_ns}
            assert offered <= receivers, f"Expected only nodes that receive a message as entrances, got {offered}."
            assert not (level.next and offered), (
                f"Expected a graph with the next nodes {level.next} to offer no entrance of its own, got {offered}."
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
TestThreadOnSqlite = pytest.mark.xdist_group("sqlite_state_machine")(SqliteThreadMachine.TestCase)


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


@settings(max_examples=500, deadline=None)
@given(spec=graph_specs((), 0), origin=st.text())
@example(
    spec=GraphSpec(
        nodes=(
            NodeSpec(
                "route",
                FunctionSpec(RouterSpec((), (1,), (), 0), is_async=False, pauses=0, late_pauses=0, resource=None),
                defer=False,
                targets=(),
                to_end=False,
            ),
            NodeSpec(
                "work",
                FunctionSpec(WriterSpec(), is_async=False, pauses=0, late_pauses=0, resource=None),
                defer=False,
                targets=(),
                to_end=False,
            ),
            NodeSpec(
                "late",
                FunctionSpec(WriterSpec(), is_async=False, pauses=0, late_pauses=0, resource=None),
                defer=True,
                targets=(),
                to_end=False,
            ),
        ),
        entries=(0, 2),
        joins=(),
    ),
    origin="",
)
def test_run_without_a_checkpointer_follows_the_order_rules(spec: GraphSpec, origin: str) -> None:
    """
    Arrange: a drawn graph whose nodes record their executions in a trace.
    Act: run the graph without a checkpointer.
    Assert: the trace follows each order rule, and no rule reads a second run of the engine.
    """
    trace, output = run_plain(spec, origin)

    check_order_rules(spec, trace, output.get("visits", []))


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

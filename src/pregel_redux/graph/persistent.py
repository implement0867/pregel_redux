from collections.abc import Mapping
from dataclasses import dataclass
from typing import Never
from uuid import UUID, uuid4

from pregel_redux.channels import extract_writes
from pregel_redux.checkpoints import (
    ChannelVersions,
    Checkpoint,
    Checkpointer,
    JoinArrivals,
    JoinVersion,
    PendingSend,
    TaskAddress,
    TriggerVersions,
    dump_value,
    dump_values,
    load_value,
    load_values,
)
from pregel_redux.commands import Send
from pregel_redux.graph.compiled import CompiledGraph, Journal, NodeCall
from pregel_redux.pregel import (
    BranchCell,
    BranchKey,
    ChannelKey,
    Channels,
    FieldKey,
    JoinCell,
    JoinKey,
    Position,
    SendsCell,
    SendsKey,
    TriggerKey,
)

type AnyGraph = CompiledGraph[object, Never, object]


def dump_trigger_versions(versions: Mapping[TriggerKey, int]) -> TriggerVersions:
    return TriggerVersions(
        branches={key.node: version for key, version in versions.items() if isinstance(key, BranchKey)},
        joins=tuple(
            JoinVersion(sources=tuple(sorted(key.sources)), target=key.target, version=version)
            for key, version in versions.items()
            if isinstance(key, JoinKey)
        ),
    )


def load_trigger_versions(versions: TriggerVersions) -> dict[TriggerKey, int]:
    return {
        **{BranchKey(node): version for node, version in versions.branches.items()},
        **{JoinKey(frozenset(join.sources), join.target): join.version for join in versions.joins},
    }


def dump_position(graph: AnyGraph, position: Position, checkpoint_id: UUID, parent: TaskAddress | None) -> Checkpoint:
    channels = position.channels
    versions = position.versions
    return Checkpoint(
        id=checkpoint_id,
        parent=parent,
        values=dump_values(graph.channels, channels.values),
        branches=tuple(sorted(node for node, cell in channels.branches.items() if cell.triggered)),
        joins=tuple(
            JoinArrivals(sources=tuple(sorted(key.sources)), target=key.target, arrived=tuple(sorted(cell.arrived)))
            for key, cell in channels.joins.items()
            if cell.arrived
        ),
        sends=tuple(
            PendingSend(
                node=send.node,
                node_input=dump_value(
                    graph.nodes[send.node].input_adapter, graph.nodes[send.node].build_send_input(send)
                ),
            )
            for send in channels.sends.sends
        ),
        versions=ChannelVersions(
            fields={key.name: version for key, version in versions.items() if isinstance(key, FieldKey)},
            triggers=dump_trigger_versions(
                {key: version for key, version in versions.items() if isinstance(key, BranchKey | JoinKey)}
            ),
            sends=versions.get(SendsKey()),
        ),
        seen={node: dump_trigger_versions(seen) for node, seen in position.seen.items()},
    )


def load_position(graph: AnyGraph, checkpoint: Checkpoint) -> Position:
    arrivals = {JoinKey(frozenset(join.sources), join.target): frozenset(join.arrived) for join in checkpoint.joins}
    start = graph.start(load_values(graph.channels, checkpoint.values), {}).channels
    unknown_nodes = sorted(
        {*checkpoint.branches, *checkpoint.seen, *(send.node for send in checkpoint.sends)} - graph.nodes.keys()
    )
    unknown_joins = [key for key in arrivals if key not in start.joins]
    if unknown_nodes or unknown_joins:
        raise ValueError(
            f"the checkpoint refers to nodes {unknown_nodes!r} and joins {unknown_joins!r}, which are not in this graph"
        )
    channels = Channels(
        values=start.values,
        branches={node: BranchCell(triggered=node in checkpoint.branches) for node in start.branches},
        joins={key: JoinCell(key.sources, arrivals.get(key, frozenset())) for key in start.joins},
        sends=SendsCell(
            tuple(
                Send(send.node, load_value(graph.nodes[send.node].input_adapter, send.node_input))
                for send in checkpoint.sends
            )
        ),
    )
    versions: dict[ChannelKey, int] = {
        **{FieldKey(name): version for name, version in checkpoint.versions.fields.items()},
        **load_trigger_versions(checkpoint.versions.triggers),
        **({} if checkpoint.versions.sends is None else {SendsKey(): checkpoint.versions.sends}),
    }
    seen = {node: load_trigger_versions(seen) for node, seen in checkpoint.seen.items()}
    return Position(channels, versions, seen)


@dataclass(frozen=True)
class ThreadScope:
    graph: AnyGraph
    checkpointer: Checkpointer
    thread_id: str
    parent: TaskAddress | None

    async def save(self, position: Position) -> UUID:
        checkpoint_id = uuid4()
        await self.checkpointer.put(self.thread_id, dump_position(self.graph, position, checkpoint_id, self.parent))
        return checkpoint_id

    async def begin(self, position: Position) -> "tuple[Position, ThreadJournal]":
        return position, ThreadJournal(self, await self.save(position))

    async def resume(self) -> "tuple[Position, ThreadJournal] | None":
        latest = await self.checkpointer.get_latest(self.thread_id, self.parent)
        return None if latest is None else (load_position(self.graph, latest), ThreadJournal(self, latest.id))


@dataclass
class ThreadJournal:
    scope: ThreadScope
    checkpoint_id: UUID

    async def save(self, position: Position) -> None:
        self.checkpoint_id = await self.scope.save(position)

    async def enter_child(self, child: AnyGraph, call: NodeCall) -> tuple[Position, Journal]:
        address = TaskAddress(checkpoint_id=self.checkpoint_id, node=call.name, send_index=call.send_index)
        scope = ThreadScope(child, self.scope.checkpointer, self.scope.thread_id, address)
        return await scope.resume() or await scope.begin(child.start(extract_writes(call.node_input), {}))


@dataclass(frozen=True)
class PersistentGraph[StateT, InputT, OutputT]:
    graph: CompiledGraph[StateT, InputT, OutputT]
    checkpointer: Checkpointer

    async def ainvoke(self, input: InputT, *, thread_id: str, step_limit: int | None = None) -> OutputT:
        latest = await self.checkpointer.get_latest(thread_id, None)
        given = {} if latest is None else load_values(self.graph.channels, latest.values)
        scope = ThreadScope(self.graph, self.checkpointer, thread_id, None)
        position, journal = await scope.begin(self.graph.start(given, extract_writes(input)))
        return await self.graph.run(position, step_limit, journal)

    async def aresume(self, *, thread_id: str, step_limit: int | None = None) -> OutputT:
        resumed = await ThreadScope(self.graph, self.checkpointer, thread_id, None).resume()
        if resumed is None:
            raise LookupError(f"thread {thread_id!r} has no checkpoint")
        position, journal = resumed
        return await self.graph.run(position, step_limit, journal)

    def invoke(self, input: InputT, *, thread_id: str, step_limit: int | None = None) -> OutputT:
        return self.graph.run_in_new_loop(
            lambda: self.ainvoke(input, thread_id=thread_id, step_limit=step_limit), "ainvoke"
        )

    def resume(self, *, thread_id: str, step_limit: int | None = None) -> OutputT:
        return self.graph.run_in_new_loop(lambda: self.aresume(thread_id=thread_id, step_limit=step_limit), "aresume")

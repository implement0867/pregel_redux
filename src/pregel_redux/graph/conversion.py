import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Never, assert_never
from uuid import UUID

from pydantic import JsonValue

from pregel_redux.channels import (
    Stored,
    dump_json,
    load_json,
    reject_unknown_channels,
)
from pregel_redux.checkpoints import (
    ChannelValues,
    ChannelVersions,
    Checkpoint,
    CheckpointSource,
    JoinVersion,
    PendingSend,
    SavedArrival,
    SavedBranch,
    SavedJoin,
    SavedWrites,
    TriggerVersions,
)
from pregel_redux.commands import Endpoint, Send
from pregel_redux.graph.compiled import CompiledGraph
from pregel_redux.pregel import (
    Arrival,
    BranchChannel,
    BranchKey,
    ChannelKey,
    Channels,
    EphemeralValue,
    FieldKey,
    JoinChannel,
    JoinKey,
    LastValueAfterFinish,
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
    NodeName,
    Position,
    StartKey,
    TasksKey,
    TaskWrites,
    Topic,
    TriggerKey,
)

type AnyGraph = CompiledGraph[object, Never, object]


def dump_trigger_versions(versions: Mapping[StartKey | TriggerKey, int]) -> TriggerVersions:
    return TriggerVersions(
        branches={key.node: version for key, version in versions.items() if isinstance(key, BranchKey)},
        joins=tuple(
            JoinVersion(sources=tuple(sorted(key.sources)), target=key.target, version=version)
            for key, version in versions.items()
            if isinstance(key, JoinKey)
        ),
    )


def load_trigger_versions(versions: TriggerVersions) -> dict[StartKey | TriggerKey, int]:
    return {
        **{BranchKey(node): version for node, version in versions.branches.items()},
        **{JoinKey(frozenset(join.sources), join.target): join.version for join in versions.joins},
    }


def dump_send(graph: AnyGraph, send: Send) -> PendingSend:
    node = graph.nodes[send.node]
    return PendingSend(node=send.node, node_input=dump_json(node.input_adapter, node.build_send_input(send)))


def load_send(graph: AnyGraph, send: PendingSend) -> Send:
    return Send(send.node, load_json(graph.nodes[send.node].input_adapter, send.node_input))


def dump_updates(graph: AnyGraph, updates: Mapping[str, object]) -> dict[str, JsonValue]:
    return {name: dump_json(graph.channels[name].update_adapter, value) for name, value in updates.items()}


def load_updates(graph: AnyGraph, updates: Mapping[str, JsonValue]) -> dict[str, object]:
    reject_unknown_channels(graph.channels, updates)
    return {name: load_json(graph.channels[name].update_adapter, value) for name, value in updates.items()}


def dump_writes(graph: AnyGraph, writes: TaskWrites) -> SavedWrites:
    return SavedWrites(
        fields=dump_updates(graph, writes.fields),
        branches=tuple(sorted(writes.branches)),
        arrivals=tuple(
            SavedArrival(sources=tuple(sorted(arrival.join.sources)), target=arrival.join.target, source=arrival.source)
            for arrival in writes.arrivals
        ),
        sends=tuple(dump_send(graph, send) for send in writes.sends),
    )


def load_writes(graph: AnyGraph, saved: SavedWrites) -> TaskWrites:
    return TaskWrites(
        fields=load_updates(graph, saved.fields),
        branches=frozenset(saved.branches),
        arrivals=tuple(
            Arrival(JoinKey(frozenset(arrival.sources), arrival.target), arrival.source) for arrival in saved.arrivals
        ),
        sends=tuple(load_send(graph, send) for send in saved.sends),
    )


def dump_branch(node: str, channel: BranchChannel) -> SavedBranch | None:
    match channel:
        case EphemeralValue(stored=Stored()):
            return SavedBranch(node=node, finished=None)
        case LastValueAfterFinish(stored=Stored()):
            return SavedBranch(node=node, finished=channel.finished)
        case _:
            return None


def load_branch(channel: BranchChannel, saved: SavedBranch) -> BranchChannel:
    match channel:
        case EphemeralValue():
            return EphemeralValue(Stored(None))
        case LastValueAfterFinish():
            return LastValueAfterFinish(Stored(None), finished=bool(saved.finished))
        case _:
            assert_never(channel)


def dump_join(key: JoinKey, channel: JoinChannel) -> SavedJoin:
    match channel:
        case NamedBarrierValue():
            finished = None
        case NamedBarrierValueAfterFinish():
            finished = channel.finished
        case _:
            assert_never(channel)
    return SavedJoin(
        sources=tuple(sorted(key.sources)), target=key.target, seen=tuple(sorted(channel.seen)), finished=finished
    )


def load_join(channel: JoinChannel, saved: SavedJoin) -> JoinChannel:
    match channel:
        case NamedBarrierValue():
            return NamedBarrierValue(channel.names, frozenset(saved.seen))
        case NamedBarrierValueAfterFinish():
            return NamedBarrierValueAfterFinish(channel.names, frozenset(saved.seen), finished=bool(saved.finished))
        case _:
            assert_never(channel)


@dataclass(frozen=True)
class CheckpointStamp:
    checkpoint_id: UUID
    parent_id: UUID | None
    created_at: datetime


def make_checkpoint_id(parent_id: UUID | None) -> UUID:
    milliseconds = time.time_ns() // 1_000_000
    layout = (milliseconds << 80) | (7 << 76) | (secrets.randbits(12) << 64) | (2 << 62) | secrets.randbits(62)
    candidate = UUID(int=layout)
    return candidate if parent_id is None or candidate > parent_id else UUID(int=parent_id.int + 1)


def dump_position(
    graph: AnyGraph, position: Position, stamp: CheckpointStamp, checkpoint_ns: str, step: int, source: CheckpointSource
) -> Checkpoint:
    channels = position.channels
    versions = position.versions
    branches = (dump_branch(node, channel) for node, channel in channels.branches.items())
    return Checkpoint(
        id=stamp.checkpoint_id,
        parent_id=stamp.parent_id,
        ts=stamp.created_at,
        checkpoint_ns=checkpoint_ns,
        step=step,
        source=source,
        channel_values=ChannelValues(
            start=None if channels.start.stored is None else dump_updates(graph, channels.start.stored.value),
            fields={name: saved.value for name, channel in channels.fields.items() if (saved := channel.checkpoint())},
            branches=tuple(branch for branch in branches if branch is not None),
            joins=tuple(dump_join(key, channel) for key, channel in channels.joins.items() if channel.seen),
            tasks=tuple(dump_send(graph, send) for send in channels.tasks.values),
        ),
        channel_versions=ChannelVersions(
            start=versions.get(StartKey()),
            fields={key.name: version for key, version in versions.items() if isinstance(key, FieldKey)},
            triggers=dump_trigger_versions(
                {key: version for key, version in versions.items() if isinstance(key, BranchKey | JoinKey)}
            ),
            tasks=versions.get(TasksKey()),
        ),
        versions_seen={
            node: dump_trigger_versions(seen) for node, seen in position.seen.items() if isinstance(node, str)
        },
        start_seen=position.seen.get(Endpoint.start, {}).get(StartKey()),
    )


def load_position(graph: AnyGraph, checkpoint: Checkpoint) -> Position:
    values = checkpoint.channel_values
    reject_unknown_channels(graph.channels, values.fields)
    opened = graph.open({}).channels
    joins = {JoinKey(frozenset(join.sources), join.target): join for join in values.joins}
    unknown_nodes = sorted(
        {*(branch.node for branch in values.branches), *checkpoint.versions_seen, *(send.node for send in values.tasks)}
        - graph.nodes.keys()
    )
    unknown_joins = [key for key in joins if key not in opened.joins]
    if unknown_nodes or unknown_joins:
        raise ValueError(
            f"the checkpoint refers to nodes {unknown_nodes!r} and joins {unknown_joins!r}, which are not in this graph"
        )
    channels = Channels(
        start=EphemeralValue(None if values.start is None else Stored(load_updates(graph, values.start))),
        fields={
            **opened.fields,
            **{name: opened.fields[name].from_checkpoint(saved) for name, saved in values.fields.items()},
        },
        branches={
            **opened.branches,
            **{branch.node: load_branch(opened.branches[branch.node], branch) for branch in values.branches},
        },
        joins={**opened.joins, **{key: load_join(opened.joins[key], join) for key, join in joins.items()}},
        tasks=Topic(tuple(load_send(graph, send) for send in values.tasks)),
    )
    versions: dict[ChannelKey, int] = {
        **({} if checkpoint.channel_versions.start is None else {StartKey(): checkpoint.channel_versions.start}),
        **{FieldKey(name): version for name, version in checkpoint.channel_versions.fields.items()},
        **load_trigger_versions(checkpoint.channel_versions.triggers),
        **({} if checkpoint.channel_versions.tasks is None else {TasksKey(): checkpoint.channel_versions.tasks}),
    }
    seen: dict[NodeName, Mapping[StartKey | TriggerKey, int]] = {
        **{node: load_trigger_versions(seen) for node, seen in checkpoint.versions_seen.items()},
        **({} if checkpoint.start_seen is None else {Endpoint.start: {StartKey(): checkpoint.start_seen}}),
    }
    return Position(channels, versions, seen)

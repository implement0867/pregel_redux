from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, assert_never

from pregel_redux.channels import Channel, FieldChannel, Stored, reject_unknown_channels
from pregel_redux.commands import Endpoint, Send


@dataclass(frozen=True)
class StartKey:
    pass


@dataclass(frozen=True)
class FieldKey:
    name: str


@dataclass(frozen=True)
class BranchKey:
    node: str


@dataclass(frozen=True)
class JoinKey:
    sources: frozenset[str]
    target: str


@dataclass(frozen=True)
class TasksKey:
    pass


type TriggerKey = BranchKey | JoinKey
type ChannelKey = StartKey | FieldKey | TriggerKey | TasksKey
type NodeName = str | Literal[Endpoint.start]


@dataclass(frozen=True)
class EphemeralValue[ValueT]:
    stored: Stored[ValueT] | None

    def is_available(self) -> bool:
        return self.stored is not None

    def update(self, values: Sequence[ValueT]) -> "EphemeralValue[ValueT] | None":
        if values:
            return EphemeralValue(Stored(values[-1]))
        return None if self.stored is None else EphemeralValue(None)

    def consume(self) -> "EphemeralValue[ValueT] | None":
        return None

    def finish(self) -> "EphemeralValue[ValueT] | None":
        return None


@dataclass(frozen=True)
class LastValueAfterFinish[ValueT]:
    stored: Stored[ValueT] | None
    finished: bool

    def is_available(self) -> bool:
        return self.stored is not None and self.finished

    def update(self, values: Sequence[ValueT]) -> "LastValueAfterFinish[ValueT] | None":
        return LastValueAfterFinish(Stored(values[-1]), finished=False) if values else None

    def consume(self) -> "LastValueAfterFinish[ValueT] | None":
        return LastValueAfterFinish(None, finished=False) if self.finished else None

    def finish(self) -> "LastValueAfterFinish[ValueT] | None":
        if self.finished or self.stored is None:
            return None
        return LastValueAfterFinish(self.stored, finished=True)


def add_names(names: frozenset[str], seen: frozenset[str], values: Sequence[str]) -> frozenset[str] | None:
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"{unknown!r} are not sources of the join of {sorted(names)!r}")
    added = seen.union(values)
    return None if added == seen else added


@dataclass(frozen=True)
class NamedBarrierValue:
    names: frozenset[str]
    seen: frozenset[str]

    def is_available(self) -> bool:
        return self.seen == self.names

    def update(self, values: Sequence[str]) -> "NamedBarrierValue | None":
        seen = add_names(self.names, self.seen, values)
        return None if seen is None else NamedBarrierValue(self.names, seen)

    def consume(self) -> "NamedBarrierValue | None":
        return NamedBarrierValue(self.names, frozenset()) if self.is_available() else None

    def finish(self) -> "NamedBarrierValue | None":
        return None


@dataclass(frozen=True)
class NamedBarrierValueAfterFinish:
    names: frozenset[str]
    seen: frozenset[str]
    finished: bool

    def is_available(self) -> bool:
        return self.finished and self.seen == self.names

    def update(self, values: Sequence[str]) -> "NamedBarrierValueAfterFinish | None":
        seen = add_names(self.names, self.seen, values)
        return None if seen is None else NamedBarrierValueAfterFinish(self.names, seen, self.finished)

    def consume(self) -> "NamedBarrierValueAfterFinish | None":
        if not self.is_available():
            return None
        return NamedBarrierValueAfterFinish(self.names, frozenset(), finished=False)

    def finish(self) -> "NamedBarrierValueAfterFinish | None":
        if self.finished or self.seen != self.names:
            return None
        return NamedBarrierValueAfterFinish(self.names, self.seen, finished=True)


@dataclass(frozen=True)
class Topic[ItemT]:
    values: tuple[ItemT, ...]

    def update(self, values: Sequence[ItemT]) -> "Topic[ItemT] | None":
        return Topic(tuple(values)) if values or self.values else None


type BranchChannel = EphemeralValue[None] | LastValueAfterFinish[None]
type JoinChannel = NamedBarrierValue | NamedBarrierValueAfterFinish


@dataclass(frozen=True)
class Channels:
    start: EphemeralValue[Mapping[str, object]]
    fields: Mapping[str, FieldChannel]
    branches: Mapping[str, BranchChannel]
    joins: Mapping[JoinKey, JoinChannel]
    tasks: Topic[Send]

    def is_available(self, key: TriggerKey) -> bool:
        match key:
            case BranchKey():
                return self.branches[key.node].is_available()
            case JoinKey():
                return self.joins[key].is_available()
            case _:
                assert_never(key)

    def read_values(self) -> dict[str, object]:
        return {name: stored.value for name, channel in self.fields.items() if (stored := channel.get()) is not None}


@dataclass(frozen=True)
class Position:
    channels: Channels
    versions: Mapping[ChannelKey, int]
    seen: Mapping[NodeName, Mapping[StartKey | TriggerKey, int]]

    def is_triggered(self, node: str, triggers: Sequence[TriggerKey]) -> bool:
        seen = self.seen.get(node, {})
        return any(self.channels.is_available(key) and self.versions[key] > seen.get(key, 0) for key in triggers)


@dataclass(frozen=True)
class StartTask:
    writes: Mapping[str, object]


@dataclass(frozen=True)
class PullTask:
    node: str
    triggers: tuple[TriggerKey, ...]


@dataclass(frozen=True)
class PushTask:
    index: int
    send: Send


type Task = StartTask | PullTask | PushTask


@dataclass(frozen=True)
class Arrival:
    join: JoinKey
    source: str


@dataclass(frozen=True)
class TaskWrites:
    fields: Mapping[str, object]
    branches: frozenset[str]
    arrivals: tuple[Arrival, ...]
    sends: tuple[Send, ...]


def open_channels(
    fields: Mapping[str, Channel],
    given: Mapping[str, object],
    branches: Mapping[str, BranchChannel],
    joins: Mapping[JoinKey, JoinChannel],
) -> Position:
    reject_unknown_channels(fields, given)
    opened = {name: channel.open() for name, channel in fields.items()}
    channels = Channels(
        start=EphemeralValue(None),
        fields={**opened, **{name: opened[name].replace(value) for name, value in given.items()}},
        branches=branches,
        joins=joins,
        tasks=Topic(()),
    )
    return Position(channels, {FieldKey(name): 1 for name in channels.read_values()}, {})


def plan_tasks(position: Position, triggers: Mapping[str, Sequence[TriggerKey]]) -> list[Task]:
    stored = position.channels.start.stored
    seen_start = position.seen.get(Endpoint.start, {}).get(StartKey(), 0)
    starts = [StartTask(stored.value)] if stored is not None and position.versions[StartKey()] > seen_start else []
    pulls = [
        PullTask(node, tuple(triggers[node]))
        for node in sorted(triggers)
        if position.is_triggered(node, triggers[node])
    ]
    return [*starts, *pulls, *(PushTask(index, send) for index, send in enumerate(position.channels.tasks.values))]


def change_channels[KeyT, ChannelT](
    channels: Mapping[KeyT, ChannelT], change: Callable[[KeyT, ChannelT], ChannelT | None]
) -> tuple[dict[KeyT, ChannelT], set[KeyT]]:
    changes = {key: changed for key, channel in channels.items() if (changed := change(key, channel)) is not None}
    return {**channels, **changes}, set(changes)


def group_values[KeyT, ValueT](pairs: Sequence[tuple[KeyT, ValueT]]) -> dict[KeyT, list[ValueT]]:
    groups: dict[KeyT, list[ValueT]] = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return groups


def finish_channels(position: Position) -> Position:
    channels = position.channels
    branches, finished_branches = change_channels(channels.branches, lambda node, channel: channel.finish())
    joins, finished_joins = change_channels(channels.joins, lambda key, channel: channel.finish())
    changed: list[ChannelKey] = [*(BranchKey(node) for node in finished_branches), *finished_joins]
    next_version = max(position.versions.values(), default=0) + 1
    return Position(
        Channels(channels.start, channels.fields, branches, joins, channels.tasks),
        {**position.versions, **dict.fromkeys(changed, next_version)},
        position.seen,
    )


def apply_writes(
    position: Position,
    tasks: Sequence[Task],
    writes: Sequence[TaskWrites],
    inputs: Sequence[Mapping[str, object]],
    trigger_keys: Collection[TriggerKey],
) -> Position:
    channels = position.channels
    pulls = [task for task in tasks if isinstance(task, PullTask)]
    bump_step = bool(tasks)
    seen: dict[NodeName, Mapping[StartKey | TriggerKey, int]] = {
        **position.seen,
        **(
            {Endpoint.start: {StartKey(): position.versions[StartKey()]}}
            if any(isinstance(task, StartTask) for task in tasks)
            else {}
        ),
        **{
            task.node: {
                **position.seen.get(task.node, {}),
                **{key: position.versions[key] for key in task.triggers if key in position.versions},
            }
            for task in pulls
        },
    }
    consumed = {key for task in pulls for key in task.triggers}

    field_values = group_values([(name, value) for write in writes for name, value in write.fields.items()])
    reject_unknown_channels(channels.fields, field_values)
    branch_values = group_values([(node, None) for write in writes for node in write.branches])
    join_values = group_values([(arrival.join, arrival.source) for write in writes for arrival in write.arrivals])
    sends = [send for write in writes for send in write.sends]

    start = channels.start.update(inputs) if inputs or (bump_step and channels.start.is_available()) else None
    fields, changed_fields = change_channels(
        channels.fields, lambda name, channel: channel.update(field_values.get(name, ()))
    )
    consumed_branches, branches_changed_by_consume = change_channels(
        channels.branches, lambda node, channel: channel.consume() if BranchKey(node) in consumed else None
    )
    branches, changed_branches = change_channels(
        consumed_branches,
        lambda node, channel: (
            channel.update(branch_values.get(node, ()))
            if node in branch_values or (bump_step and channel.is_available())
            else None
        ),
    )
    consumed_joins, joins_changed_by_consume = change_channels(
        channels.joins, lambda key, channel: channel.consume() if key in consumed else None
    )
    joins, changed_joins = change_channels(
        consumed_joins, lambda key, channel: channel.update(join_values.get(key, ()))
    )
    tasks_channel = channels.tasks.update(sends) if sends or bump_step else None

    start_updated = start is not None and start.is_available()
    updated_triggers: set[TriggerKey] = {
        *(BranchKey(node) for node in changed_branches if branches[node].is_available()),
        *(key for key in changed_joins if joins[key].is_available()),
    }
    finished_branches: set[str] = set()
    finished_joins: set[JoinKey] = set()
    if bump_step and not start_updated and updated_triggers.isdisjoint(trigger_keys):
        branches, finished_branches = change_channels(branches, lambda node, channel: channel.finish())
        joins, finished_joins = change_channels(joins, lambda key, channel: channel.finish())

    changed: list[ChannelKey] = [
        *([] if start is None else [StartKey()]),
        *(FieldKey(name) for name in changed_fields),
        *(BranchKey(node) for node in branches_changed_by_consume | changed_branches | finished_branches),
        *(joins_changed_by_consume | changed_joins | finished_joins),
        *([] if tasks_channel is None else [TasksKey()]),
    ]
    next_version = max(position.versions.values(), default=0) + 1
    return Position(
        Channels(
            start=channels.start if start is None else start,
            fields=fields,
            branches=branches,
            joins=joins,
            tasks=channels.tasks if tasks_channel is None else tasks_channel,
        ),
        {**position.versions, **dict.fromkeys(changed, next_version)},
        seen,
    )

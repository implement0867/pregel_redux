from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import assert_never

from pregel_redux.channels import Channel, apply_writes
from pregel_redux.commands import Send


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
class SendsKey:
    pass


type TriggerKey = BranchKey | JoinKey
type ChannelKey = FieldKey | TriggerKey | SendsKey


@dataclass(frozen=True)
class BranchCell:
    triggered: bool

    def advance(self, *, written: bool) -> "BranchCell | None":
        return BranchCell(written) if written or self.triggered else None


@dataclass(frozen=True)
class JoinCell:
    sources: frozenset[str]
    arrived: frozenset[str]

    def is_available(self) -> bool:
        return self.arrived == self.sources

    def advance(self, arrivals: Collection[str]) -> "JoinCell | None":
        arrived = self.arrived.union(arrivals)
        return None if arrived == self.arrived else JoinCell(self.sources, arrived)

    def consume(self) -> "JoinCell":
        return JoinCell(self.sources, frozenset())


@dataclass(frozen=True)
class SendsCell:
    sends: tuple[Send, ...]

    def advance(self, sends: Sequence[Send]) -> "SendsCell | None":
        return SendsCell(tuple(sends)) if sends or self.sends else None


@dataclass(frozen=True)
class Channels:
    values: Mapping[str, object]
    branches: Mapping[str, BranchCell]
    joins: Mapping[JoinKey, JoinCell]
    sends: SendsCell

    def is_available(self, key: TriggerKey) -> bool:
        match key:
            case BranchKey():
                return self.branches[key.node].triggered
            case JoinKey():
                return self.joins[key].is_available()
            case _:
                assert_never(key)


@dataclass(frozen=True)
class Position:
    channels: Channels
    versions: Mapping[ChannelKey, int]
    seen: Mapping[str, Mapping[TriggerKey, int]]

    def find_fired(self, node: str, triggers: Sequence[TriggerKey]) -> tuple[TriggerKey, ...]:
        seen = self.seen.get(node, {})
        return tuple(
            key for key in triggers if self.channels.is_available(key) and self.versions[key] > seen.get(key, 0)
        )


@dataclass(frozen=True)
class PullTask:
    node: str
    triggers: tuple[TriggerKey, ...]


@dataclass(frozen=True)
class PushTask:
    index: int
    send: Send


type Task = PullTask | PushTask


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


def start_position(values: Mapping[str, object], nodes: Collection[str], joins: Collection[JoinKey]) -> Position:
    channels = Channels(
        values=values,
        branches={node: BranchCell(triggered=False) for node in nodes},
        joins={key: JoinCell(key.sources, frozenset()) for key in joins},
        sends=SendsCell(()),
    )
    return Position(channels, {FieldKey(name): 1 for name in values}, {})


def plan_tasks(position: Position, triggers: Mapping[str, Sequence[TriggerKey]]) -> list[Task]:
    pulls = [PullTask(node, fired) for node in sorted(triggers) if (fired := position.find_fired(node, triggers[node]))]
    return [*pulls, *(PushTask(index, send) for index, send in enumerate(position.channels.sends.sends))]


def replace_cells[KeyT, CellT](
    cells: Mapping[KeyT, CellT], change: Callable[[KeyT, CellT], CellT | None]
) -> tuple[dict[KeyT, CellT], list[KeyT]]:
    changes = {key: changed for key, cell in cells.items() if (changed := change(key, cell)) is not None}
    return {**cells, **changes}, list(changes)


def advance(
    fields: Mapping[str, Channel], position: Position, tasks: Sequence[Task], writes: Sequence[TaskWrites]
) -> Position:
    pulls = [task for task in tasks if isinstance(task, PullTask)]
    seen = {
        **position.seen,
        **{
            task.node: {**position.seen.get(task.node, {}), **{key: position.versions[key] for key in task.triggers}}
            for task in pulls
        },
    }
    consumed = {key for task in pulls for key in task.triggers if isinstance(key, JoinKey)}
    written_branches = frozenset[str]().union(*(write.branches for write in writes))
    arrivals = [arrival for write in writes for arrival in write.arrivals]
    branches, changed_branches = replace_cells(
        position.channels.branches, lambda node, cell: cell.advance(written=node in written_branches)
    )
    joins, changed_joins = replace_cells(
        {**position.channels.joins, **{key: position.channels.joins[key].consume() for key in consumed}},
        lambda key, cell: cell.advance([arrival.source for arrival in arrivals if arrival.join == key]),
    )
    sends = position.channels.sends.advance([send for write in writes for send in write.sends])
    channels = Channels(
        values=apply_writes(fields, position.channels.values, [write.fields for write in writes]),
        branches=branches,
        joins=joins,
        sends=position.channels.sends if sends is None else sends,
    )
    changed: list[ChannelKey] = [
        *(FieldKey(name) for write in writes for name in write.fields),
        *(BranchKey(node) for node in changed_branches),
        *consumed,
        *changed_joins,
        *([] if sends is None else [SendsKey()]),
    ]
    next_version = max(position.versions.values(), default=0) + 1
    return Position(channels, {**position.versions, **dict.fromkeys(changed, next_version)}, seen)

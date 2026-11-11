from typing import TypedDict

import pytest

from pregel_redux import Send
from pregel_redux.channels import build_channels
from pregel_redux.pregel import (
    Arrival,
    BranchChannel,
    BranchKey,
    ChannelKey,
    EphemeralValue,
    FieldKey,
    JoinChannel,
    JoinKey,
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
    Position,
    PullTask,
    TasksKey,
    TaskWrites,
    apply_writes,
    open_channels,
    plan_tasks,
)


class State(TypedDict, total=False):
    topic: str
    count: int


def build_join() -> JoinKey:
    return JoinKey(frozenset({"first", "second"}), "target")


def open_position() -> Position:
    branches: dict[str, BranchChannel] = {node: EphemeralValue(None) for node in ("first", "second", "target")}
    joins: dict[JoinKey, JoinChannel] = {build_join(): NamedBarrierValue(build_join().sources, frozenset())}
    return open_channels(build_channels(State), {"topic": "dogs"}, branches, joins)


@pytest.mark.parametrize(
    ("writes", "changed"),
    [
        (TaskWrites({"topic": "cats"}, frozenset(), (), ()), {FieldKey("topic")}),
        (TaskWrites({}, frozenset({"first"}), (), ()), {BranchKey("first")}),
        (TaskWrites({}, frozenset(), (Arrival(build_join(), "first"),), ()), {build_join()}),
        (TaskWrites({}, frozenset(), (), (Send("first", {}),)), {TasksKey()}),
        (
            TaskWrites({"count": 1}, frozenset({"first", "second"}), (), ()),
            {FieldKey("count"), BranchKey("first"), BranchKey("second")},
        ),
    ],
)
def test_only_the_channels_that_a_step_changed_get_the_next_version(
    writes: TaskWrites, changed: set[ChannelKey]
) -> None:
    """
    Arrange: a start position, and the writes of one task.
    Act: apply the writes.
    Assert: the changed channels have the next version, and all other versions are the same.
    """
    start = open_position()

    actual = apply_writes(start, [], [writes], [], [BranchKey("first")]).versions

    expected = {**start.versions, **dict.fromkeys(changed, 2)}
    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize("steps_without_a_write", [1, 2, 3])
def test_node_is_planned_one_time_for_one_write_to_its_branch(steps_without_a_write: int) -> None:
    """
    Arrange: a position where one task wrote to the branch of a node.
    Act: plan the tasks, and apply steps in which no task writes.
    Assert: only the first plan has a task for the node.
    """
    triggers = {"first": (BranchKey("first"),)}
    position = apply_writes(open_position(), [], [TaskWrites({}, frozenset({"first"}), (), ())], [], [])

    plans = [plan_tasks(position, triggers)]
    for _ in range(steps_without_a_write):
        position = apply_writes(position, plans[-1], [], [], [BranchKey("first")])
        plans.append(plan_tasks(position, triggers))

    expected = [[PullTask("first", (BranchKey("first"),))], *[[]] * steps_without_a_write]
    assert plans == expected, f"Expected {expected}, got {plans}. A seen version must not start the node again."


@pytest.mark.parametrize(
    "channel",
    [
        NamedBarrierValue(frozenset({"first", "second"}), frozenset()),
        NamedBarrierValueAfterFinish(frozenset({"first", "second"}), frozenset(), finished=False),
    ],
)
def test_join_channel_rejects_a_name_that_is_not_one_of_its_sources(channel: JoinChannel) -> None:
    """
    Arrange: a join channel of two sources.
    Act: update the channel with the name of another node.
    Assert: the update raises a ValueError that names the node.
    """
    with pytest.raises(ValueError, match="other"):
        channel.update(["first", "other"])

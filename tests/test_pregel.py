from typing import TypedDict

import pytest

from pregel_redux import Send
from pregel_redux.channels import build_channels
from pregel_redux.pregel import (
    Arrival,
    BranchKey,
    ChannelKey,
    FieldKey,
    JoinKey,
    PullTask,
    SendsKey,
    TaskWrites,
    advance,
    plan_tasks,
    start_position,
)


class State(TypedDict, total=False):
    topic: str
    count: int


def build_join() -> JoinKey:
    return JoinKey(frozenset({"first", "second"}), "target")


@pytest.mark.parametrize(
    ("writes", "changed"),
    [
        (TaskWrites({"topic": "cats"}, frozenset(), (), ()), {FieldKey("topic")}),
        (TaskWrites({}, frozenset({"first"}), (), ()), {BranchKey("first")}),
        (TaskWrites({}, frozenset(), (Arrival(build_join(), "first"),), ()), {build_join()}),
        (TaskWrites({}, frozenset(), (), (Send("first", {}),)), {SendsKey()}),
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
    start = start_position({"topic": "dogs"}, ["first", "second", "target"], [build_join()])

    actual = advance(build_channels(State), start, [], [writes]).versions

    expected = {**start.versions, **dict.fromkeys(changed, 2)}
    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize("steps_without_a_write", [1, 2, 3])
def test_node_is_planned_one_time_for_one_write_to_its_branch(steps_without_a_write: int) -> None:
    """
    Arrange: a position where one task wrote to the branch of a node.
    Act: plan the tasks, and apply steps in which no task writes.
    Assert: only the first plan has a task for the node.
    """
    fields = build_channels(State)
    triggers = {"first": (BranchKey("first"),)}
    start = start_position({}, ["first"], [])
    position = advance(fields, start, [], [TaskWrites({}, frozenset({"first"}), (), ())])

    plans = [plan_tasks(position, triggers)]
    for _ in range(steps_without_a_write):
        position = advance(fields, position, plans[-1], [])
        plans.append(plan_tasks(position, triggers))

    expected = [[PullTask("first", (BranchKey("first"),))], *[[]] * steps_without_a_write]
    assert plans == expected, f"Expected {expected}, got {plans}. A seen version must not start the node again."

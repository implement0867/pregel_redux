import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, StrEnum
from typing import Annotated, Literal, TypedDict
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ConfigDict, Field, Json, TypeAdapter, computed_field
from typing_extensions import TypeForm

from pregel_redux import CheckpointSource, InMemoryCheckpointer
from pregel_redux.checkpoints import (
    Checkpoint,
    CompletedTask,
    EdgeTask,
    InterruptedTask,
    SavedCheckpoint,
    SendTask,
    TaskAddress,
    dump_value,
)


class Color(Enum):
    red = "red"
    blue = "blue"


class Point(BaseModel):
    x: int
    y: int


class Alias(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    name: str = Field(alias="title")


class Report(BaseModel):
    title: str = Field(serialization_alias="headline")


class Subject(BaseModel):
    topic: str = Field(validation_alias="subject")


class NameOnly(BaseModel):
    model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)
    topic: str = Field(alias="subject")


class Raw(BaseModel):
    data: Json[list[int]]


class Area(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int

    @computed_field
    @property
    def area(self) -> int:
        return self.width * 2


class Limits(BaseModel):
    name: str
    limit: int = 10


class Stamped(BaseModel):
    name: str
    token: UUID = Field(default_factory=uuid4)


class Shape(BaseModel):
    kind: str


class Circle(Shape):
    radius: int


@dataclass
class Counter:
    name: str
    count: int = field(init=False, default=0)


class Request(TypedDict):
    topic: str
    tags: tuple[str, ...]


class ShapeKind(StrEnum):
    circle = "circle"
    square = "square"


class Circles(BaseModel):
    kind: Literal[ShapeKind.circle]
    radius: int


class Squares(BaseModel):
    kind: Literal[ShapeKind.square]
    side: int


def build_checkpoint(thread_id: str, caller: TaskAddress | None = None) -> Checkpoint:
    return Checkpoint(
        thread_id=thread_id,
        caller=caller,
        id=uuid4(),
        parent_id=None,
        created_at=datetime.now(UTC),
        step=0,
        source=CheckpointSource.input,
        values={"count": "1.5", "items": '["a", {"b": null}]'},
        tasks=(EdgeTask(node="first"), SendTask(node="second", node_input='{"topic": "cats"}')),
    )


async def collect(checkpoints: AsyncIterator[SavedCheckpoint]) -> list[UUID]:
    return [saved.checkpoint.id async for saved in checkpoints]


def build_counter() -> Counter:
    counter = Counter(name="clicks")
    counter.count = 3
    return counter


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (datetime, datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)),
        (timedelta, timedelta(hours=1, microseconds=5)),
        (tuple[int, str], (1, "a")),
        (set[int], {1, 2}),
        (frozenset[str], frozenset({"a"})),
        (Decimal, Decimal("1.50")),
        (bytes, b"data"),
        (UUID, UUID(int=7)),
        (Color, Color.blue),
        (Point, Point(x=1, y=2)),
        (Alias, Alias(title="cats")),
        (Counter, Counter(name="clicks")),
        (Request, {"topic": "cats", "tags": ("a",)}),
        (Annotated[Circles | Squares, Field(discriminator="kind")], Squares(kind=ShapeKind.square, side=2)),
        (dict[str, list[Point]], {"points": [Point(x=1, y=2)]}),
        (int | str, "1"),
        (int | None, None),
        (datetime, datetime(2026, 3, 28, 12, tzinfo=timezone(timedelta(hours=2)))),
        (Report, Report(title="cats")),
        (Subject, Subject.model_validate({"subject": "cats"})),
        (NameOnly, NameOnly.model_validate({"topic": "cats"})),
        (Raw, Raw.model_validate({"data": "[1, 2]"})),
        (Area, Area(width=2)),
        (Limits, Limits(name="cats")),
    ],
    ids=lambda item: getattr(item, "__name__", None),
)
def test_value_of_its_declared_type_is_restored_equal(value_type: TypeForm[object], value: object) -> None:
    adapter = TypeAdapter[object](value_type)

    record = SendTask(node="node", node_input=dump_value(adapter, value, "the value"))
    restored = adapter.validate_json(SendTask.model_validate_json(record.model_dump_json()).node_input, by_name=True)

    assert restored == value, f"Expected {value!r} after a round trip through JSON, got {restored!r}."


@pytest.mark.parametrize("value", [Limits(name="cats"), Limits(name="cats", limit=10), Limits(name="dogs", limit=3)])
def test_restored_model_keeps_the_fields_that_were_set(value: Limits) -> None:
    adapter = TypeAdapter[object](Limits)

    restored = adapter.validate_json(dump_value(adapter, value, "the value"), by_name=True)

    assert isinstance(restored, Limits), f"Expected a Limits value, got {restored!r}."
    actual = restored.model_fields_set
    assert actual == value.model_fields_set, f"Expected the set fields {value.model_fields_set}, got {actual}."


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (Shape, Circle(kind="circle", radius=1)),
        (list[Shape], [Circle(kind="circle", radius=1)]),
        (Counter, build_counter()),
        (list[int] | tuple[int, ...], (1, 2)),
        (float, float("inf")),
        (float, float("nan")),
        (object, Point(x=1, y=2)),
        (object, object()),
        (datetime, datetime(2026, 3, 28, 12, tzinfo=ZoneInfo("Europe/Paris"))),
        (object, ShapeKind.circle),
        (dict[str, object], {"kind": ShapeKind.circle}),
        (Stamped, Stamped(name="cats")),
    ],
    ids=[
        "subclass",
        "subclass-in-a-list",
        "field-without-init",
        "later-union-member",
        "infinity",
        "nan",
        "model-as-object",
        "plain-object",
        "time-zone-with-rules",
        "string-enum-as-object",
        "string-enum-in-a-dict",
        "unset-field-with-a-default-factory",
    ],
)
def test_value_that_would_not_be_restored_equal_is_rejected(value_type: TypeForm[object], value: object) -> None:
    adapter = TypeAdapter[object](value_type)

    with pytest.raises(ValueError):
        dump_value(adapter, value, "the value")


def test_saved_checkpoint_is_read_back_equal() -> None:
    checkpointer = InMemoryCheckpointer()
    checkpoint = build_checkpoint("thread")
    address = TaskAddress(checkpoint_id=checkpoint.id, index=1)
    completed = CompletedTask(writes={"items": "[1.5]"}, targets=("third",), sends=())
    interrupted = InterruptedTask(answers=("true", "null"), payload={"question": [1, "two", float("inf")]})

    async def save_and_read() -> SavedCheckpoint | None:
        await checkpointer.put_checkpoint(checkpoint)
        await checkpointer.put_task_state("thread", address, interrupted)
        await checkpointer.put_task_state("thread", address.model_copy(update={"index": 0}), completed)
        return await checkpointer.read_checkpoint("thread", checkpoint.id)

    actual = asyncio.run(save_and_read())

    expected = SavedCheckpoint(checkpoint=checkpoint, task_states={0: completed, 1: interrupted})
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_checkpoints_are_listed_newest_first_for_one_caller_of_one_thread() -> None:
    checkpointer = InMemoryCheckpointer()
    root = [build_checkpoint("thread") for _ in range(3)]
    caller = TaskAddress(checkpoint_id=root[0].id, index=0)
    child = [build_checkpoint("thread", caller) for _ in range(2)]
    other = build_checkpoint("other")

    async def save_and_list() -> tuple[list[UUID], list[UUID]]:
        for checkpoint in [root[0], *child[:1], root[1], *child[1:], other, root[2]]:
            await checkpointer.put_checkpoint(checkpoint)
        return (
            await collect(checkpointer.list_checkpoints("thread", None)),
            await collect(checkpointer.list_checkpoints("thread", caller)),
        )

    actual = asyncio.run(save_and_list())

    expected = ([item.id for item in reversed(root)], [item.id for item in reversed(child)])
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_task_state_of_an_unknown_checkpoint_is_rejected() -> None:
    checkpointer = InMemoryCheckpointer()
    state = InterruptedTask(answers=(), payload=None)

    with pytest.raises(LookupError):
        asyncio.run(checkpointer.put_task_state("thread", TaskAddress(checkpoint_id=uuid4(), index=0), state))


def test_delete_thread_removes_only_the_checkpoints_of_that_thread() -> None:
    checkpointer = InMemoryCheckpointer()
    kept = build_checkpoint("kept")
    removed = build_checkpoint("removed")

    async def save_and_delete() -> tuple[SavedCheckpoint | None, SavedCheckpoint | None]:
        await checkpointer.put_checkpoint(kept)
        await checkpointer.put_checkpoint(removed)
        await checkpointer.delete_thread("removed")
        await checkpointer.delete_thread("missing")
        return (
            await checkpointer.read_checkpoint("kept", kept.id),
            await checkpointer.read_checkpoint("removed", removed.id),
        )

    kept_saved, removed_saved = asyncio.run(save_and_delete())

    assert kept_saved is not None, "Expected the other thread to keep its checkpoint."
    assert removed_saved is None, "Expected the deleted thread to have no checkpoint."

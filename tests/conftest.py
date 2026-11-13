from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field

import pytest

from pregel_redux.snapshots import FailedTask, FinishedTask, TaskSnapshot


@dataclass
class Journal:
    events: list[str] = field(default_factory=list)
    opened: int = 0

    def open_resource(self, name: str) -> int:
        self.opened += 1
        self.events.append(f"open {name} {self.opened}")
        return self.opened


def join_lists[T](current: list[T], update: list[T]) -> list[T]:
    return current + update


def append_item[T](current: list[T], update: T) -> list[T]:
    return [*current, update]


def read_error(task: TaskSnapshot) -> str | None:
    return task.status.error if isinstance(task.status, FailedTask) else None


def read_result(task: TaskSnapshot) -> Mapping[str, object] | None:
    return task.status.result if isinstance(task.status, FinishedTask) else None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("xdist_group") is None:
            item.add_marker(pytest.mark.xdist_group("other_tests"))


@pytest.fixture
def journal() -> Journal:
    return Journal()


async def collect[ItemT](items: AsyncIterator[ItemT]) -> list[ItemT]:
    return [item async for item in items]

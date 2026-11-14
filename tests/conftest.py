from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from pregel_redux.snapshots import FailedTask, TaskSnapshot


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


def read_error(task: TaskSnapshot) -> str:
    match task.status:
        case FailedTask(error=error):
            return error
        case _:
            raise AssertionError(f"Expected a failed task, got {task}.")


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

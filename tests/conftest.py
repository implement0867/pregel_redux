from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest


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


@pytest.fixture
def journal() -> Journal:
    return Journal()


async def collect[ItemT](items: AsyncIterator[ItemT]) -> list[ItemT]:
    return [item async for item in items]

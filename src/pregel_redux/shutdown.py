import signal
import time
from collections.abc import Callable, Collection, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from types import FrameType


class ShutdownLevel(Enum):
    graceful = "graceful"
    immediate = "immediate"


class RunAborted(Exception):
    pass


@dataclass(eq=False)
class ShutdownState:
    level: ShutdownLevel | None = None
    deadline: float | None = None
    listeners: list[Callable[[], object]] = field(default_factory=list)


@dataclass(frozen=True, eq=False)
class ShutdownSignal:
    state: ShutdownState

    @property
    def requested(self) -> bool:
        return self.state.level is not None

    @property
    def aborted(self) -> bool:
        deadline = self.state.deadline
        return self.state.level is ShutdownLevel.immediate or (deadline is not None and time.monotonic() >= deadline)

    @property
    def deadline(self) -> float | None:
        return self.state.deadline

    def subscribe(self, listener: Callable[[], object]) -> Callable[[], None]:
        listeners = self.state.listeners
        listeners.append(listener)

        def unsubscribe() -> None:
            if listener in listeners:
                listeners.remove(listener)

        return unsubscribe


@dataclass(frozen=True, eq=False)
class ShutdownController:
    signal: ShutdownSignal = field(default_factory=lambda: ShutdownSignal(ShutdownState()))

    def shutdown(self, grace_period: float | None = None) -> None:
        state = self.signal.state
        if state.level is None:
            state.level = ShutdownLevel.graceful
        if grace_period is not None:
            deadline = time.monotonic() + grace_period
            state.deadline = deadline if state.deadline is None else min(state.deadline, deadline)
        for listener in list(state.listeners):
            listener()

    def abort(self) -> None:
        state = self.signal.state
        state.level = ShutdownLevel.immediate
        for listener in list(state.listeners):
            listener()


def read_shutdown() -> ShutdownSignal:
    raise RuntimeError("read_shutdown supplies a value only in a run of a graph that has a checkpointer")


@contextmanager
def graceful_shutdown(
    *,
    grace_period: float | None = None,
    signals: Collection[signal.Signals] = (signal.SIGINT, signal.SIGTERM),
) -> Generator[ShutdownSignal]:
    controller = ShutdownController()

    def handle(number: int, frame: FrameType | None) -> None:
        if controller.signal.requested:
            controller.abort()
        else:
            controller.shutdown(grace_period)

    earlier = {number: signal.signal(number, handle) for number in signals}
    try:
        yield controller.signal
    finally:
        for number, handler in earlier.items():
            signal.signal(number, handler)

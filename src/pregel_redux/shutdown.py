import asyncio
import signal
import time
from collections.abc import Callable, Collection, Coroutine, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import FrameType
from typing import Protocol, assert_never


class RunAborted(Exception):
    """Signals that the shutdown signal of a run cancelled the running nodes."""


@dataclass(frozen=True)
class NotRequested:
    pass


@dataclass(frozen=True)
class Graceful:
    deadline: float | None


@dataclass(frozen=True)
class Aborted:
    pass


type ShutdownState = NotRequested | Graceful | Aborted


class ShutdownSignal(Protocol):
    """Reports shutdown requests to a run.

    A requested shutdown stops the run before its next step.
    An abort cancels the running nodes and raises `RunAborted`.
    `seconds_until_abort` returns `None` until a shutdown request with a grace period.
    """

    @property
    def requested(self) -> bool: ...

    @property
    def aborted(self) -> bool: ...

    def seconds_until_abort(self) -> float | None: ...

    def subscribe(self, listener: Callable[[], object]) -> Callable[[], None]: ...


@dataclass(eq=False)
class ShutdownController:
    """Sends shutdown requests to each run that receives `signal`.

    Parameters
    ----------
    clock
        Returns the current time in seconds for the grace period.
    """

    clock: Callable[[], float] = time.monotonic
    state: ShutdownState = field(default_factory=NotRequested, init=False)
    listeners: list[Callable[[], object]] = field(default_factory=list, init=False)

    @property
    def signal(self) -> ShutdownSignal:
        return self

    @property
    def requested(self) -> bool:
        match self.state:
            case NotRequested():
                return False
            case Graceful() | Aborted():
                return True
            case _:
                assert_never(self.state)

    @property
    def aborted(self) -> bool:
        remaining = self.seconds_until_abort()
        return remaining is not None and remaining <= 0

    def seconds_until_abort(self) -> float | None:
        match self.state:
            case NotRequested():
                return None
            case Graceful(deadline=deadline):
                return None if deadline is None else max(0.0, deadline - self.clock())
            case Aborted():
                return 0.0
            case _:
                assert_never(self.state)

    def subscribe(self, listener: Callable[[], object]) -> Callable[[], None]:
        """Calls `listener` after each shutdown request and after each abort.

        Returns a function that removes `listener`.
        """
        listeners = self.listeners
        listeners.append(listener)

        def unsubscribe() -> None:
            if listener in listeners:
                listeners.remove(listener)

        return unsubscribe

    def shutdown(self, grace_period: float | None = None) -> None:
        """Stops each run before its next step.

        Parameters
        ----------
        grace_period
            The seconds until the abort.
            `None` lets the running nodes finish.
            Several calls keep the earliest deadline.
        """
        deadline = None if grace_period is None else self.clock() + grace_period
        match self.state:
            case NotRequested():
                self.enter(Graceful(deadline))
            case Graceful(deadline=earlier):
                deadlines = [each for each in (earlier, deadline) if each is not None]
                self.enter(Graceful(min(deadlines, default=None)))
            case Aborted():
                self.enter(self.state)
            case _:
                assert_never(self.state)

    def abort(self) -> None:
        """Cancels the running nodes of each run.

        Each run raises `RunAborted`.
        """
        self.enter(Aborted())

    def enter(self, state: ShutdownState) -> None:
        self.state = state
        for listener in list(self.listeners):
            listener()


async def run_until_abort[ResultT](shutdown: ShutdownSignal, work: Coroutine[object, object, ResultT]) -> ResultT:
    if shutdown.aborted:
        work.close()
        raise RunAborted("the shutdown signal of the run is aborted")
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(work)
    timers: list[asyncio.TimerHandle] = []
    aborting = False

    def abort() -> None:
        nonlocal aborting
        if not aborting:
            aborting = True
            task.cancel()

    def react() -> None:
        remaining = shutdown.seconds_until_abort()
        if remaining is None:
            return
        if remaining <= 0:
            abort()
        else:
            timers.append(loop.call_later(remaining, abort))

    unsubscribe = shutdown.subscribe(lambda: loop.call_soon_threadsafe(react))
    react()
    try:
        return await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if not aborting or (current is not None and current.cancelling()):
            raise
        raise RunAborted("the run was aborted by its shutdown signal") from None
    finally:
        unsubscribe()
        for timer in timers:
            timer.cancel()


@contextmanager
def graceful_shutdown(
    *,
    grace_period: float | None = None,
    signals: Collection[signal.Signals] = (signal.SIGINT, signal.SIGTERM),
) -> Generator[ShutdownSignal]:
    """Converts operating system signals to shutdown requests for the duration of the context.

    The first signal requests a shutdown.
    The second signal aborts each run.
    The context restores the earlier signal handlers at exit.

    Parameters
    ----------
    grace_period
        The seconds from the first signal until the abort.
    """
    controller = ShutdownController()

    def handle(number: int, frame: FrameType | None) -> None:
        if controller.signal.requested:
            controller.abort()
        else:
            controller.shutdown(grace_period)

    earlier: dict[signal.Signals, Callable[[int, FrameType | None], object] | int | None] = {}
    try:
        for number in signals:
            earlier[number] = signal.signal(number, handle)
        yield controller.signal
    finally:
        for number, handler in earlier.items():
            signal.signal(number, handler)

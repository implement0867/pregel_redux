import asyncio
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Generator
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from queue import SimpleQueue

from pregel_redux.channels import Stored
from pregel_redux.streams import (
    ChildPart,
    StreamPart,
)


def run_in_new_loop[ResultT](start_run: Callable[[], Coroutine[object, object, ResultT]]) -> ResultT:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        in_loop = False
    else:
        in_loop = True
    if not in_loop:
        return asyncio.run(start_run())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(copy_context().run, asyncio.run, start_run()).result()


def iterate_in_new_loop[ItemT](start: Callable[[], AsyncIterator[ItemT]]) -> Generator[ItemT]:
    items = SimpleQueue[Stored[ItemT] | None]()
    lock = threading.Lock()
    cancels: list[Callable[[], object]] = []

    async def pump() -> None:
        async for item in start():
            items.put(Stored(item))

    async def supervise() -> None:
        task = asyncio.ensure_future(pump())
        loop = asyncio.get_running_loop()
        cancels.append(lambda: loop.call_soon_threadsafe(task.cancel))
        try:
            await task
        finally:
            with lock:
                cancels.clear()

    with ThreadPoolExecutor(max_workers=1) as executor:
        work = executor.submit(copy_context().run, asyncio.run, supervise())
        work.add_done_callback(lambda _: items.put(None))
        try:
            while (item := items.get()) is not None:
                yield item.value
        finally:
            with lock:
                for cancel in cancels:
                    cancel()
            match work.exception():
                case None | asyncio.CancelledError():
                    pass
                case error:
                    raise error


async def stream_run[ResultT](
    start_run: Callable[[Callable[[StreamPart | ChildPart], None]], Coroutine[object, object, ResultT]],
) -> AsyncGenerator[StreamPart | ChildPart]:
    parts = asyncio.Queue[StreamPart | ChildPart | None]()
    loop = asyncio.get_running_loop()

    def emit(part: StreamPart | ChildPart | None) -> None:
        loop.call_soon_threadsafe(parts.put_nowait, part)

    async def work() -> ResultT:
        try:
            return await start_run(emit)
        finally:
            emit(None)

    task = asyncio.create_task(work())
    try:
        while (part := await parts.get()) is not None:
            yield part
    finally:
        task.cancel()
        await asyncio.wait([task])
        if not task.cancelled() and (error := task.exception()) is not None:
            raise error

import asyncio
import threading

import pytest

from async_utils import run_blocking


async def test_run_blocking_signals_and_waits_for_thread_on_cancellation() -> None:
    started = threading.Event()
    stop = threading.Event()
    exited = threading.Event()

    def blocking() -> None:
        started.set()
        stop.wait(timeout=1)
        exited.set()

    task = asyncio.create_task(run_blocking(blocking, cancel=stop.set))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert exited.is_set()


async def test_run_blocking_releases_caller_after_cancellation_grace() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking() -> None:
        started.set()
        release.wait(timeout=1)

    task = asyncio.create_task(run_blocking(blocking, cancel_grace_seconds=0.01))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.2)

    release.set()

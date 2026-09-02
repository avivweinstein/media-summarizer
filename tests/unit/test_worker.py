import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

from async_utils import run_blocking
from worker import JobWorker


async def test_worker_can_restart_without_stale_sentinels(mocker: MagicMock) -> None:
    run_job = mocker.patch("worker.run_job", new_callable=AsyncMock)
    worker = JobWorker(concurrency=1)

    worker.start()
    await asyncio.sleep(0)
    await worker.stop()
    assert worker.queue_size == 0

    worker.start()
    await worker.enqueue("job-1")
    await asyncio.wait_for(worker._queue.join(), timeout=1)
    await worker.stop()

    run_job.assert_awaited_once_with("job-1")


async def test_cancel_stops_active_job_without_stopping_worker(mocker: MagicMock) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    completed: list[str] = []

    async def run(job_id: str) -> None:
        if job_id == "job-1":
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        else:
            completed.append(job_id)

    mocker.patch("worker.run_job", side_effect=run)
    worker = JobWorker(concurrency=1)
    worker.start()
    await worker.enqueue("job-1")
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await worker.cancel("job-1")
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    await worker.enqueue("job-2")
    await asyncio.wait_for(worker._queue.join(), timeout=1)
    await worker.stop()

    assert completed == ["job-2"]


async def test_cancel_acknowledgement_is_bounded_for_slow_blocking_work(
    mocker: MagicMock,
) -> None:
    started = threading.Event()
    release = threading.Event()
    completed: list[str] = []

    def blocking() -> None:
        started.set()
        release.wait(timeout=1)

    async def run(job_id: str) -> None:
        if job_id == "job-1":
            await run_blocking(blocking, cancel_grace_seconds=0.01)
        else:
            completed.append(job_id)

    mocker.patch("worker.run_job", side_effect=run)
    worker = JobWorker(concurrency=1, cancel_wait_seconds=0.1)
    worker.start()
    await worker.enqueue("job-1")
    assert await asyncio.to_thread(started.wait, 1)

    assert await asyncio.wait_for(worker.cancel("job-1"), timeout=0.2)
    await worker.enqueue("job-2")
    await asyncio.wait_for(worker._queue.join(), timeout=0.2)

    release.set()
    await worker.stop()

    assert completed == ["job-2"]

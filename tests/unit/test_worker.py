import asyncio
from unittest.mock import AsyncMock, MagicMock

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

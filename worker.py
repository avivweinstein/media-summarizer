"""Async job worker with bounded concurrency.

Instead of using FastAPI BackgroundTasks (which provides no concurrency control),
this module runs a fixed-size pool of workers that pull jobs from an asyncio queue.
Long-running jobs (podcast downloads, Whisper transcription) no longer block
the API or other jobs.

Usage (in main.py lifespan):
    from worker import job_worker
    job_worker.start()
    ...
    await job_worker.stop()

    # To enqueue a job:
    await job_worker.enqueue(job_id)
"""

import asyncio
import logging

from pipeline import run_job

logger = logging.getLogger(__name__)

# How many jobs can run in parallel
DEFAULT_CONCURRENCY = 2


class JobWorker:
    """Manages a pool of async workers that process jobs from a queue."""

    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY) -> None:
        self.concurrency = concurrency
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    def start(self) -> None:
        """Spawn worker tasks. Call once during app startup."""
        if self._running:
            return
        self._running = True
        for i in range(self.concurrency):
            task = asyncio.create_task(self._worker(i), name=f"job-worker-{i}")
            self._tasks.append(task)
        logger.info(
            "job_id=- url=- source=- event=worker_pool_started concurrency=%d",
            self.concurrency,
        )

    async def stop(self) -> None:
        """Signal all workers to stop and wait for them to finish."""
        self._running = False
        # Send sentinel values to unblock workers waiting on the queue
        for _ in self._tasks:
            await self._queue.put("")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("job_id=- url=- source=- event=worker_pool_stopped")

    async def enqueue(self, job_id: str) -> None:
        """Add a job ID to the work queue."""
        await self._queue.put(job_id)
        logger.info("job_id=%s url=- source=- event=job_enqueued queue_size=%d", job_id, self._queue.qsize())

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def _worker(self, worker_id: int) -> None:
        """Worker loop: pull job IDs from the queue and process them."""
        while self._running:
            try:
                job_id = await self._queue.get()
                if not job_id:  # sentinel
                    break
                logger.info(
                    "job_id=%s url=- source=- event=worker_picked_up worker=%d",
                    job_id, worker_id,
                )
                await run_job(job_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "job_id=- url=- source=- event=worker_error worker=%d error=%r",
                    worker_id, str(e),
                )
            finally:
                self._queue.task_done()


# Singleton instance used by the app
job_worker = JobWorker()

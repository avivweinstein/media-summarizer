"""Cancellation-safe wrappers for blocking work used by the job pipeline."""

import asyncio
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
DEFAULT_CANCEL_GRACE_SECONDS = 1.0


def _consume_task_result(task: asyncio.Task[object]) -> None:
    try:
        task.result()
    except BaseException:
        pass


async def run_blocking(
    function: Callable[..., T],
    *args: object,
    cancel: Callable[[], None] | None = None,
    cancel_grace_seconds: float = DEFAULT_CANCEL_GRACE_SECONDS,
) -> T:
    """Signal blocking work on cancellation without retaining a worker indefinitely."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if cancel is not None:
            cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=cancel_grace_seconds)
        except TimeoutError:
            task.add_done_callback(_consume_task_result)
        except Exception:
            pass
        raise

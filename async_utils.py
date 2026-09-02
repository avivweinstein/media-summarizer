"""Cancellation-safe wrappers for blocking work used by the job pipeline."""

import asyncio
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


async def run_blocking(
    function: Callable[..., T],
    *args: object,
    cancel: Callable[[], None] | None = None,
) -> T:
    """Keep ownership until executor work exits, signalling cancellation when supported."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if cancel is not None:
            cancel()
        try:
            await asyncio.shield(task)
        except Exception:
            pass
        raise

"""Abstract base for all media sources."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from models import TranscriptResult, UsageStats


class BaseSource(ABC):
    @abstractmethod
    async def fetch(
        self,
        url: str,
        job_id: str = "-",
        *,
        usage: UsageStats | None = None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None = None,
        processing_mode: str = "cloud_public",
    ) -> TranscriptResult:
        """Fetch transcript and metadata for the given URL."""
        ...

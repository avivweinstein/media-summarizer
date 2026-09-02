"""Abstract base for all media sources."""

from abc import ABC, abstractmethod

from models import TranscriptResult, UsageStats


class BaseSource(ABC):
    @abstractmethod
    async def fetch(
        self,
        url: str,
        job_id: str = "-",
        *,
        usage: UsageStats | None = None,
    ) -> TranscriptResult:
        """Fetch transcript and metadata for the given URL."""
        ...

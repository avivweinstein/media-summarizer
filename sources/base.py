"""Abstract base for all media sources."""

from abc import ABC, abstractmethod

from models import TranscriptResult


class BaseSource(ABC):
    @abstractmethod
    async def fetch(self, url: str) -> TranscriptResult:
        """Fetch transcript and metadata for the given URL."""
        ...

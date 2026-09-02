"""Custom exceptions for the media summarizer pipeline."""


class MediaSummarizerError(Exception):
    """Base exception for all pipeline errors."""


class UsageLimitError(MediaSummarizerError):
    """A configured request, size, duration, or cost limit was reached."""


class UnsupportedURLError(MediaSummarizerError):
    """URL pattern not recognised or explicitly unsupported (e.g. Spotify)."""


class NoTranscriptError(MediaSummarizerError):
    """YouTube video has no available transcript. No audio fallback."""


class MetadataError(MediaSummarizerError):
    """Failed to fetch video/episode metadata."""


class SummarizationError(MediaSummarizerError):
    """Claude API call failed or returned unparseable output."""


class NotionError(MediaSummarizerError):
    """Notion API call failed."""


class ObsidianError(MediaSummarizerError):
    """Obsidian vault validation or note writing failed."""


class TranscriptionError(MediaSummarizerError):
    """Whisper transcription failed."""

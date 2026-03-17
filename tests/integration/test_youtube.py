"""YouTube integration tests — hit the real YouTube transcript and metadata APIs."""

import pytest

from models import TranscriptResult
from sources.youtube import YouTubeSource

pytestmark = pytest.mark.integration

# "Me at the zoo" — first YouTube video ever, 19 seconds, stable, has captions
_SHORT_VIDEO_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


async def test_fetch_returns_transcript_result(youtube_api_key: str) -> None:
    source = YouTubeSource(youtube_api_key=youtube_api_key)
    result = await source.fetch(_SHORT_VIDEO_URL, job_id="integration-test")

    assert isinstance(result, TranscriptResult)
    assert result.source == "youtube"


async def test_transcript_is_non_empty(youtube_api_key: str) -> None:
    source = YouTubeSource(youtube_api_key=youtube_api_key)
    result = await source.fetch(_SHORT_VIDEO_URL, job_id="integration-test")

    assert len(result.transcript) > 10


async def test_title_is_populated(youtube_api_key: str) -> None:
    source = YouTubeSource(youtube_api_key=youtube_api_key)
    result = await source.fetch(_SHORT_VIDEO_URL, job_id="integration-test")

    assert result.title
    assert len(result.title) > 0


async def test_channel_is_populated(youtube_api_key: str) -> None:
    source = YouTubeSource(youtube_api_key=youtube_api_key)
    result = await source.fetch(_SHORT_VIDEO_URL, job_id="integration-test")

    assert result.channel_or_show


async def test_duration_is_positive(youtube_api_key: str) -> None:
    source = YouTubeSource(youtube_api_key=youtube_api_key)
    result = await source.fetch(_SHORT_VIDEO_URL, job_id="integration-test")

    assert result.duration_seconds > 0

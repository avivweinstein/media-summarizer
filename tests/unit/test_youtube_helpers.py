"""Tests for YouTube source helper functions.

Pure-function tests — no network calls, no mocks needed.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config import settings
from exceptions import MetadataError, UsageLimitError
from models import TranscriptionOutput, TranscriptSegment
from sources.youtube import (
    YouTubeSource,
    _download_audio_sync,
    _extract_video_id,
    _HostRestrictedYoutubeDL,
    _parse_upload_date,
)


def test_host_restricted_extractor_blocks_request_before_network(mocker: MagicMock) -> None:
    network = mocker.patch("sources.youtube.yt_dlp.YoutubeDL.urlopen", return_value="ok")
    downloader = _HostRestrictedYoutubeDL({}, frozenset({"x.com"}))

    with pytest.raises(MetadataError, match="untrusted host"):
        downloader.urlopen("https://internal.example/private")
    assert downloader.urlopen("https://x.com/example/status/1") == "ok"

    network.assert_called_once()


async def test_native_transcript_preserves_segments(mocker: MagicMock) -> None:
    mocker.patch(
        "sources.youtube._fetch_metadata_sync",
        return_value={"title": "Timestamped", "duration": 120},
    )
    mocker.patch(
        "sources.youtube._fetch_transcript_sync",
        return_value=TranscriptionOutput(
            text="A useful detail.",
            segments=[
                TranscriptSegment(
                    start_seconds=30,
                    end_seconds=35,
                    text="A useful detail.",
                )
            ],
        ),
    )

    result = await YouTubeSource().fetch("https://youtube.com/watch?v=timestamped")

    assert result.transcript == "A useful detail."
    assert result.segments[0].start_seconds == 30


async def test_duration_limit_blocks_before_youtube_download(
    mocker: MagicMock,
) -> None:
    mocker.patch(
        "sources.youtube._fetch_metadata_sync",
        return_value={"title": "Long video", "duration": 120},
    )
    mocker.patch("sources.youtube._fetch_transcript_sync", return_value=None)
    download = mocker.patch("sources.youtube._download_audio_sync")
    mocker.patch.object(settings, "max_audio_duration_seconds", 60)

    with pytest.raises(UsageLimitError, match="duration"):
        await YouTubeSource().fetch("https://youtube.com/watch?v=long")

    download.assert_not_called()


def test_youtube_download_hook_stops_at_size_limit(
    tmp_path: Path,
    mocker: MagicMock,
) -> None:
    class FakeYoutubeDL:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def download(self, _urls: list[str]) -> int:
            hooks = self.options["progress_hooks"]
            assert isinstance(hooks, list)
            hooks[0]({"downloaded_bytes": 6})
            return 0

    mocker.patch("sources.youtube.yt_dlp.YoutubeDL", FakeYoutubeDL)
    destination = tmp_path / "job.mp3"

    with pytest.raises(UsageLimitError, match="download-size"):
        _download_audio_sync("https://youtube.com/watch?v=large", destination, 5)

    assert not list(tmp_path.glob("job.*"))


class TestExtractVideoId:
    def test_standard_watch_url(self) -> None:
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_watch_url_with_extra_params(self) -> None:
        assert _extract_video_id("https://www.youtube.com/watch?v=abc123&t=30s") == "abc123"

    def test_youtu_be_url(self) -> None:
        assert _extract_video_id("https://youtu.be/abc123") == "abc123"

    def test_youtu_be_with_query_params(self) -> None:
        assert _extract_video_id("https://youtu.be/abc123?t=30") == "abc123"

    def test_no_v_param_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            _extract_video_id("https://www.youtube.com/channel/UCabc")

    def test_empty_youtu_be_path_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            _extract_video_id("https://youtu.be/")


class TestParseUploadDate:
    def test_valid_date(self) -> None:
        result = _parse_upload_date("20050424")
        assert result == datetime(2005, 4, 24, tzinfo=UTC)

    def test_another_valid_date(self) -> None:
        result = _parse_upload_date("20231015")
        assert result == datetime(2023, 10, 15, tzinfo=UTC)

    def test_none_returns_none(self) -> None:
        assert _parse_upload_date(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_upload_date("") is None

    def test_wrong_length_returns_none(self) -> None:
        assert _parse_upload_date("2023") is None
        assert _parse_upload_date("202310151200") is None

    def test_invalid_month_returns_none(self) -> None:
        assert _parse_upload_date("20231315") is None

    def test_invalid_day_returns_none(self) -> None:
        assert _parse_upload_date("20231032") is None

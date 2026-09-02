from unittest.mock import AsyncMock, MagicMock

import pytest

from exceptions import MetadataError, UnsupportedURLError
from models import TranscriptionOutput, TranscriptSegment
from sources.media import MediaSource, _fetch_twitter_metadata_sync, _vimeo_player_url


def test_standard_vimeo_url_uses_embedded_player() -> None:
    assert _vimeo_player_url("https://vimeo.com/123") == "https://player.vimeo.com/video/123"
    assert _vimeo_player_url("https://www.vimeo.com/123") == "https://player.vimeo.com/video/123"


async def test_generic_media_downloads_and_preserves_timestamps(mocker: MagicMock) -> None:
    mocker.patch("sources.media._validate_public_http_url", new=AsyncMock(return_value="1.1.1.1"))
    mocker.patch(
        "sources.media._fetch_metadata_sync",
        return_value={
            "id": "vimeo-123",
            "title": "Conference Talk",
            "duration": 300,
            "uploader": "Example Conference",
        },
    )
    mocker.patch("sources.media._download_audio_sync")
    mocker.patch(
        "sources.media.transcribe",
        new=AsyncMock(
            return_value=TranscriptionOutput(
                text="A conference transcript.",
                segments=[TranscriptSegment(start_seconds=10, end_seconds=15, text="A detail.")],
            )
        ),
    )

    result = await MediaSource().fetch("https://vimeo.com/123", job_id="job")

    assert result.source == "media"
    assert result.title == "Conference Talk"
    assert result.segments[0].start_seconds == 10


async def test_direct_media_uses_pinned_streaming_download(mocker: MagicMock) -> None:
    download = mocker.patch("sources.media._download_mp3", new=AsyncMock())
    convert = mocker.patch("sources.media.convert_to_mp3", new=AsyncMock())
    transcribe = mocker.patch(
        "sources.media.transcribe",
        new=AsyncMock(return_value=TranscriptionOutput(text="Direct media transcript.")),
    )

    result = await MediaSource().fetch("https://cdn.example.com/talk.mp4", job_id="job")

    download.assert_awaited_once()
    assert download.call_args.args[1].suffix == ".mp4"
    convert.assert_awaited_once()
    assert transcribe.call_args.args[0].suffix == ".mp3"
    assert result.title == "talk.mp4"


async def test_twitter_video_uses_single_item_hosted_media_path(mocker: MagicMock) -> None:
    url = "https://x.com/example/status/1234567890/video/2"
    validate = mocker.patch(
        "sources.media._validate_public_http_url", new=AsyncMock(return_value="1.1.1.1")
    )
    metadata_result = {
        "id": "1234567890",
        "title": "Example post",
        "duration": 30,
        "uploader": "Example User",
        "timestamp": 1_700_000_000,
        "formats": [{"url": "https://video.twimg.com/example.m3u8"}],
    }
    metadata = mocker.patch(
        "sources.media._fetch_twitter_metadata_sync",
        return_value={
            **metadata_result,
        },
    )
    download = mocker.patch("sources.media._download_audio_sync")
    mocker.patch(
        "sources.media.transcribe",
        new=AsyncMock(return_value=TranscriptionOutput(text="X video transcript.")),
    )

    result = await MediaSource().fetch(url, job_id="twitter-job")

    validate.assert_awaited_once_with(url)
    metadata.assert_called_once_with(url)
    assert download.call_args.args[4] is True
    assert download.call_args.args[5] == metadata_result
    assert result.source == "twitter"
    assert result.source_item_id == "1234567890"
    assert result.channel_or_show == "Example User"
    assert result.published_at is not None
    assert result.published_at.isoformat() == "2023-11-14T22:13:20+00:00"


async def test_twitter_video_reports_unavailable_or_gated_post(mocker: MagicMock) -> None:
    mocker.patch(
        "sources.media._validate_public_http_url", new=AsyncMock(return_value="1.1.1.1")
    )
    mocker.patch(
        "sources.media._fetch_twitter_metadata_sync",
        side_effect=MetadataError("unavailable or requires login"),
    )

    with pytest.raises(MetadataError, match="requires login"):
        await MediaSource().fetch("https://x.com/example/status/1234567890", job_id="job")


def test_twitter_metadata_rejects_bare_multi_video_post(mocker: MagicMock) -> None:
    extractor = MagicMock()
    extractor.__enter__.return_value.extract_info.return_value = {
        "_type": "playlist",
        "entries": [
            {"formats": [{"url": "https://video.twimg.com/one.m3u8"}]},
            {"formats": [{"url": "https://video.twimg.com/two.m3u8"}]},
        ],
    }
    extractor.__enter__.return_value.process_ie_result.side_effect = (
        lambda info, download: info
    )
    mocker.patch("sources.media.yt_dlp.YoutubeDL", return_value=extractor)

    with pytest.raises(MetadataError, match="multiple videos"):
        _fetch_twitter_metadata_sync("https://x.com/example/status/1234567890")


@pytest.mark.parametrize(
    "raw",
    [
        {"_type": "url", "url": "http://127.0.0.1/internal"},
        {"formats": [{"url": "https://untrusted.example/video.mp4"}]},
    ],
)
def test_twitter_metadata_rejects_external_delegation(
    mocker: MagicMock, raw: dict[str, object]
) -> None:
    extractor = MagicMock()
    extractor.__enter__.return_value.extract_info.return_value = raw
    extractor.__enter__.return_value.process_ie_result.side_effect = (
        lambda info, download: info
    )
    mocker.patch("sources.media.yt_dlp.YoutubeDL", return_value=extractor)

    with pytest.raises(MetadataError, match="native video|untrusted"):
        _fetch_twitter_metadata_sync("https://x.com/example/status/1234567890")


async def test_twitter_age_gated_video_is_rejected(mocker: MagicMock) -> None:
    mocker.patch(
        "sources.media._validate_public_http_url", new=AsyncMock(return_value="1.1.1.1")
    )
    mocker.patch(
        "sources.media._fetch_twitter_metadata_sync",
        return_value={
            "age_limit": 18,
            "formats": [{"url": "https://video.twimg.com/example.m3u8"}],
        },
    )
    download = mocker.patch("sources.media._download_audio_sync")

    with pytest.raises(UnsupportedURLError, match="Age-gated"):
        await MediaSource().fetch("https://x.com/example/status/1234567890", job_id="job")

    download.assert_not_called()

from unittest.mock import AsyncMock, MagicMock

from models import TranscriptionOutput, TranscriptSegment
from sources.media import MediaSource, _vimeo_player_url


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

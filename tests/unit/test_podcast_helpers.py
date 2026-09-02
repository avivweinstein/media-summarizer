"""Tests for pure-function helpers in sources/podcast.py."""

import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config import settings
from exceptions import UnsupportedURLError, UsageLimitError
from models import TranscriptionOutput
from sources.podcast import (
    PodcastSource,
    _best_mp3_entry,
    _download_mp3,
    _episode_source_item_id,
    _fetch_rss,
    _parse_apple_podcast_ids,
    _parse_duration,
    _struct_to_datetime,
    _thumbnail_from_entry,
    _validate_public_http_url,
)


async def test_apple_episode_uses_exact_itunes_track_lookup(
    mocker: MagicMock,
) -> None:
    episode_id = "1000694698631"
    lookup = mocker.patch(
        "sources.podcast._apple_episode_metadata",
        return_value={
            "trackId": int(episode_id),
            "episodeUrl": "https://cdn.example.com/exact.mp3",
            "trackName": "Exact Episode",
            "collectionName": "Test Show",
            "trackTimeMillis": 120_000,
            "releaseDate": "2026-01-02T03:04:05Z",
        },
    )
    download = mocker.patch("sources.podcast._download_mp3")
    mocker.patch(
        "sources.podcast.transcribe",
        return_value=TranscriptionOutput(text="Transcript"),
    )

    result = await PodcastSource()._from_apple_podcasts(
        f"https://podcasts.apple.com/us/podcast/show/id123?i={episode_id}",
        "job",
        None,
        None,
    )

    lookup.assert_awaited_once_with(
        f"https://podcasts.apple.com/us/podcast/show/id123?i={episode_id}",
        episode_id,
    )
    assert download.call_args.args[0] == "https://cdn.example.com/exact.mp3"
    assert result.title == "Exact Episode"
    assert result.source_item_id == episode_id


async def test_rss_fetch_stops_at_response_size_limit(mocker: MagicMock) -> None:
    class FakeResponse:
        headers: dict[str, str] = {}

        async def __aenter__(self) -> "FakeResponse":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self, _size: int) -> AsyncIterator[bytes]:
            yield b"123"
            yield b"456"

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def stream(self, *_args: object, **_kwargs: object) -> FakeResponse:
            return FakeResponse()

    mocker.patch("sources.podcast.httpx.AsyncClient", return_value=FakeClient())
    mocker.patch(
        "sources.podcast._validate_public_http_url",
        return_value="93.184.216.34",
    )
    mocker.patch("sources.podcast._RSS_MAX_BYTES", 5)

    with pytest.raises(UsageLimitError, match="response-size"):
        await _fetch_rss("https://example.com/opaque-feed")


async def test_download_stops_and_cleans_up_at_size_limit(
    tmp_path: Path, mocker: MagicMock
) -> None:
    class FakeResponse:
        headers: dict[str, str] = {}

        async def __aenter__(self) -> "FakeResponse":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self, _size: int) -> AsyncIterator[bytes]:
            for chunk in (b"123", b"456"):
                yield chunk

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def stream(self, *_args: object, **_kwargs: object) -> FakeResponse:
            return FakeResponse()

    destination = tmp_path / "too-large.mp3"
    mocker.patch("sources.podcast.httpx.AsyncClient", return_value=FakeClient())
    mocker.patch(
        "sources.podcast._validate_public_http_url",
        return_value="93.184.216.34",
    )
    mocker.patch.object(settings, "max_audio_download_bytes", 5)

    with pytest.raises(UsageLimitError, match="download-size"):
        await _download_mp3("https://example.com/audio.mp3", destination)

    assert not destination.exists()


async def test_private_media_url_is_rejected() -> None:
    with pytest.raises(UnsupportedURLError, match="Private"):
        await _validate_public_http_url("http://127.0.0.1/feed.xml")


async def test_rss_redirect_to_private_address_is_rejected(
    mocker: MagicMock,
) -> None:
    class RedirectResponse:
        status_code = 302
        headers = {"location": "http://127.0.0.1/private.xml"}

        async def __aenter__(self) -> "RedirectResponse":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def stream(self, *_args: object, **_kwargs: object) -> RedirectResponse:
            return RedirectResponse()

    mocker.patch("sources.podcast.httpx.AsyncClient", return_value=FakeClient())

    with pytest.raises(UnsupportedURLError, match="Private"):
        await _fetch_rss("http://93.184.216.34/feed")


def test_episode_source_item_id_prefers_guid() -> None:
    entry = {"id": "episode-guid"}

    assert _episode_source_item_id(entry, "https://cdn.example.com/audio.mp3") == ("episode-guid")


def test_episode_source_item_id_falls_back_to_enclosure() -> None:
    assert _episode_source_item_id({}, "https://cdn.example.com/audio.mp3") == (
        "https://cdn.example.com/audio.mp3"
    )


class TestParseApplePodcastIds:
    def test_show_only(self) -> None:
        url = "https://podcasts.apple.com/us/podcast/some-show/id1545953110"
        podcast_id, episode_id = _parse_apple_podcast_ids(url)
        assert podcast_id == "1545953110"
        assert episode_id is None

    def test_specific_episode(self) -> None:
        url = "https://podcasts.apple.com/us/podcast/some-show/id1545953110?i=1000694698631"
        podcast_id, episode_id = _parse_apple_podcast_ids(url)
        assert podcast_id == "1545953110"
        assert episode_id == "1000694698631"

    def test_trailing_slash(self) -> None:
        url = "https://podcasts.apple.com/us/podcast/some-show/id1545953110/"
        podcast_id, _ = _parse_apple_podcast_ids(url)
        assert podcast_id == "1545953110"

    def test_no_id_raises(self) -> None:
        url = "https://podcasts.apple.com/us/podcast/some-show/"
        with pytest.raises(ValueError, match="Cannot extract podcast ID"):
            _parse_apple_podcast_ids(url)


class TestParseDuration:
    def test_plain_seconds(self) -> None:
        assert _parse_duration("3600") == 3600

    def test_mm_ss(self) -> None:
        assert _parse_duration("45:30") == 45 * 60 + 30

    def test_hh_mm_ss(self) -> None:
        assert _parse_duration("1:02:03") == 3600 + 2 * 60 + 3

    def test_integer_input(self) -> None:
        assert _parse_duration(90) == 90

    def test_none_returns_zero(self) -> None:
        assert _parse_duration(None) == 0

    def test_empty_string_returns_zero(self) -> None:
        assert _parse_duration("") == 0

    def test_invalid_string_returns_zero(self) -> None:
        assert _parse_duration("not-a-duration") == 0

    def test_zero_seconds(self) -> None:
        assert _parse_duration("0") == 0

    def test_hh_mm_ss_with_leading_zeros(self) -> None:
        assert _parse_duration("01:00:00") == 3600


class TestStructToDatetime:
    def test_valid_struct(self) -> None:
        t = time.strptime("2024-01-15", "%Y-%m-%d")
        result = _struct_to_datetime(t)
        assert result == datetime(2024, 1, 15, tzinfo=UTC)

    def test_none_returns_none(self) -> None:
        assert _struct_to_datetime(None) is None

    def test_result_is_utc_aware(self) -> None:
        t = time.strptime("2023-06-01 12:30:00", "%Y-%m-%d %H:%M:%S")
        result = _struct_to_datetime(t)
        assert result is not None
        assert result.tzinfo is UTC


class TestThumbnailFromEntry:
    def _make_obj(self, image: object) -> object:
        class Obj:
            pass

        o = Obj()
        setattr(o, "image", image)
        return o

    def test_entry_image_href(self) -> None:
        entry = self._make_obj({"href": "https://example.com/thumb.jpg"})
        feed = self._make_obj(None)
        assert _thumbnail_from_entry(entry, feed) == "https://example.com/thumb.jpg"

    def test_entry_image_url(self) -> None:
        entry = self._make_obj({"url": "https://example.com/thumb.jpg"})
        feed = self._make_obj(None)
        assert _thumbnail_from_entry(entry, feed) == "https://example.com/thumb.jpg"

    def test_falls_back_to_feed_image(self) -> None:
        entry = self._make_obj(None)
        feed = self._make_obj({"href": "https://example.com/feed-thumb.jpg"})
        assert _thumbnail_from_entry(entry, feed) == "https://example.com/feed-thumb.jpg"

    def test_returns_none_when_neither_has_image(self) -> None:
        entry = self._make_obj(None)
        feed = self._make_obj(None)
        assert _thumbnail_from_entry(entry, feed) is None

    def test_entry_takes_precedence_over_feed(self) -> None:
        entry = self._make_obj({"href": "https://example.com/entry.jpg"})
        feed = self._make_obj({"href": "https://example.com/feed.jpg"})
        assert _thumbnail_from_entry(entry, feed) == "https://example.com/entry.jpg"


def _make_feed_entry(
    title: str,
    mp3_url: str,
    guid: str = "",
    audio_type: str = "audio/mpeg",
) -> dict:  # type: ignore[type-arg]
    """Build a minimal feedparser-compatible entry dict."""
    return {
        "title": title,
        "id": guid or mp3_url,
        "enclosures": [{"href": mp3_url, "type": audio_type}],
    }


class _MockFeed:
    """Minimal feedparser-compatible feed object."""

    def __init__(self, entries: list[dict]) -> None:  # type: ignore[type-arg]
        self.entries = entries
        self.feed = {"title": "Test Show"}


def _make_feed(entries: list[dict]) -> _MockFeed:  # type: ignore[type-arg]
    return _MockFeed(entries)


class TestBestMp3Entry:
    def test_returns_first_entry_when_no_episode_id(self) -> None:
        entries = [
            _make_feed_entry("Ep 2", "https://cdn.example.com/ep2.mp3"),
            _make_feed_entry("Ep 1", "https://cdn.example.com/ep1.mp3"),
        ]
        feed = _make_feed(entries)
        mp3_url, entry = _best_mp3_entry(feed, episode_id=None)
        assert mp3_url == "https://cdn.example.com/ep2.mp3"
        assert entry["title"] == "Ep 2"

    def test_matches_by_episode_id_in_guid(self) -> None:
        entries = [
            _make_feed_entry("Ep 2", "https://cdn.example.com/ep2.mp3", guid="guid-999"),
            _make_feed_entry(
                "Ep 1",
                "https://cdn.example.com/ep1.mp3",
                guid="guid-1000694698631-extra",
            ),
        ]
        feed = _make_feed(entries)
        mp3_url, entry = _best_mp3_entry(feed, episode_id="1000694698631")
        assert mp3_url == "https://cdn.example.com/ep1.mp3"
        assert entry["title"] == "Ep 1"

    def test_exact_episode_id_must_match(self) -> None:
        entries = [
            _make_feed_entry("Ep 2", "https://cdn.example.com/ep2.mp3", guid="guid-999"),
            _make_feed_entry("Ep 1", "https://cdn.example.com/ep1.mp3", guid="guid-888"),
        ]
        feed = _make_feed(entries)
        with pytest.raises(ValueError, match="Episode ID nonexistent"):
            _best_mp3_entry(feed, episode_id="nonexistent")

    def test_skips_entries_without_audio_enclosure(self) -> None:
        entry_no_audio = {
            "title": "Not audio",
            "id": "guid-text",
            "enclosures": [{"href": "https://example.com/doc.pdf", "type": "application/pdf"}],
        }
        entry_audio = _make_feed_entry("Has Audio", "https://cdn.example.com/ep.mp3")
        feed = _make_feed([entry_no_audio, entry_audio])
        mp3_url, entry = _best_mp3_entry(feed, episode_id=None)
        assert mp3_url == "https://cdn.example.com/ep.mp3"

    def test_accepts_mp3_url_without_audio_type(self) -> None:
        entry = _make_feed_entry("Ep", "https://cdn.example.com/ep.mp3", audio_type="")
        feed = _make_feed([entry])
        mp3_url, _ = _best_mp3_entry(feed, episode_id=None)
        assert mp3_url == "https://cdn.example.com/ep.mp3"

    def test_empty_feed_raises(self) -> None:
        feed = _make_feed([])
        with pytest.raises(ValueError, match="No episodes"):
            _best_mp3_entry(feed, episode_id=None)

    def test_feed_with_no_audio_enclosures_raises(self) -> None:
        entry = {"title": "Text only", "id": "guid-x", "enclosures": []}
        feed = _make_feed([entry])
        with pytest.raises(ValueError, match="No episodes"):
            _best_mp3_entry(feed, episode_id=None)

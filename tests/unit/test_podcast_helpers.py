"""Tests for pure-function helpers in sources/podcast.py."""

import time
from datetime import UTC, datetime

import pytest

from sources.podcast import (
    _best_mp3_entry,
    _parse_apple_podcast_ids,
    _parse_duration,
    _struct_to_datetime,
    _thumbnail_from_entry,
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

    def test_falls_back_to_latest_when_episode_id_not_found(self) -> None:
        entries = [
            _make_feed_entry("Ep 2", "https://cdn.example.com/ep2.mp3", guid="guid-999"),
            _make_feed_entry("Ep 1", "https://cdn.example.com/ep1.mp3", guid="guid-888"),
        ]
        feed = _make_feed(entries)
        mp3_url, entry = _best_mp3_entry(feed, episode_id="nonexistent")
        assert mp3_url == "https://cdn.example.com/ep2.mp3"

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

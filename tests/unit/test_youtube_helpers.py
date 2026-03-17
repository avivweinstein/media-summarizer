"""Tests for YouTube source helper functions.

Pure-function tests — no network calls, no mocks needed.
"""

from datetime import UTC, datetime

import pytest

from sources.youtube import _extract_video_id, _parse_upload_date


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

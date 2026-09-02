"""Tests for URL source detection and validation.

Covers detect_source() in pipeline.py — no network calls, pure logic.
"""

import pytest

from exceptions import UnsupportedURLError
from pipeline import detect_source


class TestYouTubeURLs:
    def test_standard_watch_url(self) -> None:
        assert detect_source("https://www.youtube.com/watch?v=abc123") == "youtube"

    def test_no_www_watch_url(self) -> None:
        assert detect_source("https://youtube.com/watch?v=abc123") == "youtube"

    def test_youtu_be_url(self) -> None:
        assert detect_source("https://youtu.be/abc123") == "youtube"

    def test_mobile_youtube_url(self) -> None:
        assert detect_source("https://m.youtube.com/watch?v=abc123") == "youtube"

    def test_watch_url_with_timestamp(self) -> None:
        assert detect_source("https://www.youtube.com/watch?v=abc123&t=42s") == "youtube"

    def test_watch_url_with_playlist_param(self) -> None:
        assert detect_source("https://www.youtube.com/watch?v=abc123&list=PLxyz") == "youtube"

    def test_youtu_be_with_query_params(self) -> None:
        assert detect_source("https://youtu.be/abc123?t=30") == "youtube"

    def test_youtube_channel_rejected(self) -> None:
        with pytest.raises(UnsupportedURLError, match="video URLs|playlist"):
            detect_source("https://www.youtube.com/channel/UCabc123")

    def test_youtube_user_rejected(self) -> None:
        with pytest.raises(UnsupportedURLError, match="video URLs|playlist"):
            detect_source("https://www.youtube.com/@SomeCreator")

    def test_youtube_playlist_accepted(self) -> None:
        assert detect_source("https://www.youtube.com/playlist?list=PLabc") == "youtube_playlist"

    def test_youtube_playlist_with_list_param(self) -> None:
        assert detect_source("https://www.youtube.com/playlist?list=PLxyz123") == "youtube_playlist"

    def test_youtube_homepage_rejected(self) -> None:
        with pytest.raises(UnsupportedURLError, match="video URLs|playlist"):
            detect_source("https://www.youtube.com/")


class TestSpotify:
    def test_spotify_episode_rejected_with_message(self) -> None:
        with pytest.raises(UnsupportedURLError, match="Spotify"):
            detect_source("https://open.spotify.com/episode/4rOoJ6Egrf8K2IrywzwOMk")

    def test_spotify_show_rejected(self) -> None:
        with pytest.raises(UnsupportedURLError, match="Spotify"):
            detect_source("https://open.spotify.com/show/abc123")


class TestPodcast:
    def test_apple_podcasts_url(self) -> None:
        assert detect_source("https://podcasts.apple.com/us/podcast/foo/id123456") == "podcast"

    def test_direct_mp3_url(self) -> None:
        assert detect_source("https://feeds.example.com/episodes/ep42.mp3") == "podcast"

    def test_mp3_url_with_query_params(self) -> None:
        assert detect_source("https://cdn.example.com/ep1.mp3?token=abc") == "podcast"

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/show/feed.xml",
            "https://example.com/show.rss",
            "https://feeds.simplecast.com/abc123",
            "https://rss.example.com/show",
            "https://example.com/podcast/feed",
            "https://example.com/podcast?format=rss",
        ],
    )
    def test_generic_rss_urls(self, url: str) -> None:
        assert detect_source(url) == "podcast"


class TestAdditionalSources:
    def test_vimeo_url(self) -> None:
        assert detect_source("https://vimeo.com/123456") == "media"

    def test_direct_video_url(self) -> None:
        assert detect_source("https://cdn.example.com/talk.mp4?token=abc") == "media"

    def test_article_url(self) -> None:
        assert detect_source("https://example.com/research/new-chip") == "article"


class TestUnsupportedURLs:
    def test_opaque_feed_candidate_is_accepted_for_bounded_sniffing(self) -> None:
        assert detect_source("https://example.com/opaque-feed-id") == "article"

    def test_twitter(self) -> None:
        with pytest.raises(UnsupportedURLError):
            detect_source("https://twitter.com/user/status/12345")

    def test_empty_string(self) -> None:
        with pytest.raises(UnsupportedURLError):
            detect_source("")

    def test_plain_text(self) -> None:
        with pytest.raises(UnsupportedURLError):
            detect_source("not a url at all")

    def test_soundcloud(self) -> None:
        with pytest.raises(UnsupportedURLError):
            detect_source("https://soundcloud.com/artist/track")

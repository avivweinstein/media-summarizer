from url_identity import submission_identity


def test_youtube_identity_ignores_url_variant_and_timestamp() -> None:
    standard = submission_identity("https://youtube.com/watch?v=abc&t=42")
    short = submission_identity("https://youtu.be/abc?si=tracking")

    assert standard == ("youtube:abc", True)
    assert short == standard


def test_feed_identity_ignores_only_known_tracking_parameters() -> None:
    first = submission_identity("https://feeds.example.com/show?utm_source=email&token=abc")
    second = submission_identity("https://feeds.example.com/show?token=abc")
    changed_token = submission_identity("https://feeds.example.com/show?token=def")

    assert first == second
    assert first != changed_token
    assert first[1] is False


def test_apple_episode_is_static_but_show_is_dynamic() -> None:
    episode = submission_identity("https://podcasts.apple.com/us/podcast/show/id123?i=456")
    show = submission_identity("https://podcasts.apple.com/us/podcast/show/id123")

    assert episode == ("apple-podcast:123:episode:456", True)
    assert show[1] is False


def test_direct_audio_is_static() -> None:
    assert submission_identity("https://cdn.example.com/audio.mp3?token=abc")[1] is True


def test_vimeo_variants_share_a_static_identity() -> None:
    standard = submission_identity("https://www.vimeo.com/76979871")
    player = submission_identity("https://player.vimeo.com/video/76979871")

    assert standard == ("vimeo:76979871", True)
    assert player == standard


def test_direct_video_is_static() -> None:
    assert submission_identity("https://cdn.example.com/talk.webm?token=abc")[1] is True


def test_twitter_variants_share_a_static_identity() -> None:
    x_url = submission_identity("https://x.com/example/status/1234567890?s=20")
    twitter_url = submission_identity("https://twitter.com/other/status/1234567890")

    assert x_url == ("twitter:1234567890", True)
    assert twitter_url == x_url


def test_twitter_media_suffixes_share_the_post_identity() -> None:
    first = submission_identity("https://x.com/example/status/1234567890/video/1")
    second = submission_identity("https://x.com/example/status/1234567890/video/2")

    assert first == ("twitter:1234567890", True)
    assert second == first

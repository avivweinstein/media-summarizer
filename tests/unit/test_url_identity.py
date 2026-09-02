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
    episode = submission_identity(
        "https://podcasts.apple.com/us/podcast/show/id123?i=456"
    )
    show = submission_identity("https://podcasts.apple.com/us/podcast/show/id123")

    assert episode == ("apple-podcast:123:episode:456", True)
    assert show[1] is False


def test_direct_audio_is_static() -> None:
    assert submission_identity("https://cdn.example.com/audio.mp3?token=abc")[1] is True

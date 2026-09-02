from unittest.mock import AsyncMock, MagicMock

import pytest

from exceptions import UnsupportedURLError
from sources.article import ArticleSource


async def test_extracts_article_metadata_and_body(mocker: MagicMock) -> None:
    body = b"""
        <html><head>
          <title>Fallback</title>
          <meta property="og:title" content="A Useful Article">
          <meta name="author" content="Ada Example">
          <meta property="article:published_time" content="2026-08-30T10:00:00Z">
        </head><body><nav>Ignore navigation</nav><article>
          <h1>A Useful Article</h1>
          <p>This is the first substantive paragraph with enough useful detail to retain.</p>
          <p>This is the second substantive paragraph, providing context and evidence for readers.</p>
          <p>This final paragraph makes the extracted article comfortably longer than one hundred characters.</p>
        </article><script>ignore()</script></body></html>
    """
    mocker.patch(
        "sources.article._fetch_page",
        new=AsyncMock(return_value=("https://example.com/article", "text/html", body)),
    )

    result = await ArticleSource().fetch("https://example.com/article")

    assert result.title == "A Useful Article"
    assert result.channel_or_show == "Ada Example"
    assert result.source == "article"
    assert "substantive paragraph" in result.transcript
    assert "Ignore navigation" not in result.transcript
    assert "ignore()" not in result.transcript
    assert result.published_at is not None


async def test_rss_document_delegates_to_podcast(mocker: MagicMock) -> None:
    body = b"""<?xml version="1.0"?><rss><channel><title>Show</title><item>
      <title>Episode</title><enclosure url="https://example.com/e.mp3" type="audio/mpeg" />
    </item></channel></rss>"""
    mocker.patch(
        "sources.article._fetch_page",
        new=AsyncMock(return_value=("https://example.com/opaque", "application/xml", body)),
    )
    podcast = AsyncMock()
    mocker.patch("sources.article.PodcastSource", return_value=podcast)

    await ArticleSource().fetch("https://example.com/opaque", job_id="job")

    podcast.fetch.assert_awaited_once()


async def test_extensionless_media_delegates_to_media_source(mocker: MagicMock) -> None:
    mocker.patch(
        "sources.article._fetch_page",
        new=AsyncMock(return_value=("https://cdn.example.com/item", "audio/mpeg", b"")),
    )
    media = AsyncMock()
    mocker.patch("sources.article.MediaSource", return_value=media)

    await ArticleSource().fetch("https://cdn.example.com/item", job_id="job")

    media.fetch.assert_awaited_once()


async def test_rejects_short_non_article(mocker: MagicMock) -> None:
    mocker.patch(
        "sources.article._fetch_page",
        new=AsyncMock(return_value=("https://example.com", "text/html", b"<p>tiny</p>")),
    )

    with pytest.raises(UnsupportedURLError, match="enough readable"):
        await ArticleSource().fetch("https://example.com")

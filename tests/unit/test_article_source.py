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


async def test_article_feed_uses_latest_embedded_entry(mocker: MagicMock) -> None:
    body = b"""<?xml version="1.0"?><rss><channel><title>Engineering Notes</title><item>
      <guid>post-42</guid><title>New Design</title><author>Ada Example</author>
      <pubDate>Sun, 30 Aug 2026 10:00:00 GMT</pubDate>
      <link>https://example.com/posts/new-design</link>
      <description><![CDATA[<p>This newsletter entry contains a detailed explanation of the new design.</p>
      <p>It includes enough context, evidence, and practical guidance to make the extracted
      content useful as a durable library note without fetching another page.</p>]]></description>
    </item></channel></rss>"""
    mocker.patch(
        "sources.article._fetch_page",
        new=AsyncMock(return_value=("https://example.com/feed.xml", "application/xml", body)),
    )

    result = await ArticleSource().fetch("https://example.com/feed.xml")

    assert result.source == "article"
    assert result.title == "New Design"
    assert result.url == "https://example.com/posts/new-design"
    assert result.source_item_id == "post-42"
    assert "durable library note" in result.transcript


async def test_non_audio_enclosure_stays_an_article_feed(mocker: MagicMock) -> None:
    body = b"""<rss><channel><item><title>Illustrated Post</title>
      <enclosure url="https://example.com/image.jpg" type="image/jpeg" />
      <description>This article has enough substantive text to remain an article even though
      it also includes an image enclosure for readers and feed applications to display.</description>
    </item></channel></rss>"""
    mocker.patch(
        "sources.article._fetch_page",
        new=AsyncMock(return_value=("https://example.com/feed", "application/rss+xml", body)),
    )
    podcast = mocker.patch("sources.article.PodcastSource")

    result = await ArticleSource().fetch("https://example.com/feed")

    assert result.source == "article"
    podcast.assert_not_called()


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

"""Bounded article/newsletter extraction with RSS fallback."""

from collections.abc import Awaitable, Callable
from datetime import datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import feedparser
import httpx

from config import settings
from exceptions import UnsupportedURLError
from models import TranscriptResult, UsageStats
from sources.media import MediaSource
from sources.podcast import PodcastSource, _pinned_request_args

_MAX_REDIRECTS = 5
_BLOCKED_TAGS = {"script", "style", "nav", "footer", "form", "svg", "noscript"}
_TEXT_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "pre"}


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.author = ""
        self.published = ""
        self._title_parts: list[str] = []
        self._all_parts: list[str] = []
        self._article_parts: list[str] = []
        self._blocked_depth = 0
        self._article_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value or "" for key, value in attrs}
        tag = tag.casefold()
        if tag in _BLOCKED_TAGS:
            self._blocked_depth += 1
        if tag in {"article", "main"}:
            self._article_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = (values.get("property") or values.get("name") or "").casefold()
            content = values.get("content", "").strip()
            if key in {"og:title", "twitter:title"} and content:
                self.title = content
            elif key in {"author", "article:author", "byl"} and content:
                self.author = content
            elif key in {"article:published_time", "date", "datepublished"} and content:
                self.published = content
        if tag in _TEXT_TAGS and not self._blocked_depth:
            target = self._article_parts if self._article_depth else self._all_parts
            target.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "title":
            self._in_title = False
        if tag in {"article", "main"} and self._article_depth:
            self._article_depth -= 1
        if tag in _BLOCKED_TAGS and self._blocked_depth:
            self._blocked_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._blocked_depth:
            return
        clean = " ".join(data.split())
        if not clean:
            return
        if self._in_title:
            self._title_parts.append(clean)
        self._all_parts.append(clean)
        if self._article_depth:
            self._article_parts.append(clean)

    def result(self) -> tuple[str, str, str, str]:
        title = self.title or " ".join(self._title_parts)
        chosen = (
            self._article_parts if len(" ".join(self._article_parts)) >= 200 else self._all_parts
        )
        text = "\n".join(line.strip() for line in " ".join(chosen).split("\n") if line.strip())
        return title.strip(), self.author.strip(), self.published.strip(), text


def _published_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None


def _has_audio_enclosure(feed: feedparser.FeedParserDict) -> bool:
    for entry in feed.entries:
        for enclosure in entry.get("enclosures", []):
            enclosure_type = str(enclosure.get("type") or "").casefold()
            enclosure_url = str(enclosure.get("href") or enclosure.get("url") or "").casefold()
            if enclosure_type.startswith("audio/") or enclosure_url.endswith(".mp3"):
                return True
    return False


async def _fetch_page(url: str) -> tuple[str, str, bytes]:
    current_url = url
    async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            request_url, headers, extensions = await _pinned_request_args(current_url)
            headers["User-Agent"] = "media-summarizer/0.1 (personal local knowledge tool)"
            async with client.stream(
                "GET", request_url, headers=headers, extensions=extensions
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise UnsupportedURLError("Article redirect had no destination.")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0]
                if content_type.casefold().startswith(("audio/", "video/")):
                    return current_url, content_type.casefold(), b""
                raw_length = response.headers.get("content-length", "")
                if raw_length.isdigit() and int(raw_length) > settings.max_article_download_bytes:
                    raise UnsupportedURLError("Article exceeds the configured download-size limit.")
                body = bytearray()
                async for chunk in response.aiter_bytes(64 * 1024):
                    body.extend(chunk)
                    if len(body) > settings.max_article_download_bytes:
                        raise UnsupportedURLError(
                            "Article exceeds the configured download-size limit."
                        )
                return current_url, content_type.casefold(), bytes(body)
    raise UnsupportedURLError("Article exceeded the redirect limit.")


class ArticleSource:
    async def _from_feed_entry(
        self,
        feed: feedparser.FeedParserDict,
        final_url: str,
        job_id: str,
        usage: UsageStats | None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None,
        processing_mode: str,
    ) -> TranscriptResult:
        entry = feed.entries[0]
        entry_url = str(entry.get("link") or final_url)
        fragments = entry.get("content") or []
        raw_html = " ".join(
            str(fragment.get("value") or "") for fragment in fragments if isinstance(fragment, dict)
        )
        raw_html = raw_html or str(entry.get("summary") or "")
        parser = _ArticleParser()
        parser.feed(raw_html)
        _, _, _, text = parser.result()
        if len(text) < 100 and entry_url != final_url:
            return await self.fetch(
                entry_url,
                job_id,
                usage=usage,
                persist_usage=persist_usage,
                processing_mode=processing_mode,
            )
        if len(text) < 100:
            raise UnsupportedURLError(
                "The latest feed entry did not contain enough readable article text."
            )
        if len(text) > settings.max_transcript_chars:
            raise UnsupportedURLError("Article exceeds the configured transcript-size limit.")
        feed_title = str(feed.feed.get("title") or "")
        return TranscriptResult(
            title=str(entry.get("title") or "Untitled article"),
            source="article",
            url=entry_url,
            channel_or_show=str(entry.get("author") or feed_title),
            duration_seconds=0,
            transcript=text,
            published_at=_published_at(str(entry.get("published") or entry.get("updated") or "")),
            transcription_model="local/feed-parser",
            source_item_id=str(entry.get("id") or entry_url),
        )

    async def fetch(
        self,
        url: str,
        job_id: str = "-",
        *,
        usage: UsageStats | None = None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None = None,
        processing_mode: str = "nvidia_internal",
    ) -> TranscriptResult:
        final_url, content_type, body = await _fetch_page(url)
        if content_type.startswith(("audio/", "video/")):
            return await MediaSource().fetch(
                final_url,
                job_id,
                usage=usage,
                persist_usage=persist_usage,
                processing_mode=processing_mode,
            )
        feed = feedparser.parse(body)
        if feed.entries and _has_audio_enclosure(feed):
            return await PodcastSource().fetch(
                final_url,
                job_id,
                usage=usage,
                persist_usage=persist_usage,
                processing_mode=processing_mode,
            )
        if feed.entries:
            return await self._from_feed_entry(
                feed,
                final_url,
                job_id,
                usage,
                persist_usage,
                processing_mode,
            )
        if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
            raise UnsupportedURLError(
                "URL is neither a readable article, podcast feed, nor supported media."
            )

        charset = "utf-8"
        raw_text = body.decode(charset, errors="replace")
        if content_type == "text/plain":
            title = urlparse(final_url).path.rsplit("/", 1)[-1] or "Untitled article"
            author = ""
            published = None
            text = raw_text.strip()
        else:
            parser = _ArticleParser()
            parser.feed(raw_text)
            title, author, published_value, text = parser.result()
            title = title or urlparse(final_url).hostname or "Untitled article"
            published = _published_at(published_value)
        if len(text) < 100:
            raise UnsupportedURLError("The page did not contain enough readable article text.")
        if len(text) > settings.max_transcript_chars:
            raise UnsupportedURLError("Article exceeds the configured transcript-size limit.")
        return TranscriptResult(
            title=title,
            source="article",
            url=final_url,
            channel_or_show=author,
            duration_seconds=0,
            transcript=text,
            published_at=published,
            transcription_model="local/html-parser",
            source_item_id=final_url,
        )

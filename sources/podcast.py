"""Podcast source implementation.

Supported URL types:
  - Direct MP3 URL (.mp3 extension)
  - RSS feed URL (parsed by feedparser)
  - Apple Podcasts URL (resolved via iTunes Lookup API → RSS)

Spotify URLs are explicitly rejected with a helpful error message.

Flow for each type:
  1. Resolve URL to (mp3_url, metadata_dict)
  2. Download MP3 to /tmp/media-summarizer/{job_id}.mp3
  3. Transcribe via Whisper (see transcriber.py)
  4. Return TranscriptResult
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import feedparser
import httpx

from exceptions import TranscriptionError
from models import TranscriptResult
from sources.base import BaseSource
from transcriber import tmp_path_for_job, transcribe

logger = logging.getLogger(__name__)

_ITUNES_LOOKUP = "https://itunes.apple.com/lookup"
_DOWNLOAD_TIMEOUT = 300  # seconds — large podcasts can be slow
_DOWNLOAD_CHUNK = 64 * 1024  # 64 KB chunks


# ---------------------------------------------------------------------------
# Pure helpers — URL parsing, duration, metadata extraction
# ---------------------------------------------------------------------------

def _parse_apple_podcast_ids(url: str) -> tuple[str, str | None]:
    """Return (podcast_id, episode_id_or_None) from an Apple Podcasts URL.

    URL formats:
      .../id1545953110                   → show page, no specific episode
      .../id1545953110?i=1000694698631   → specific episode
    """
    parsed = urlparse(url)
    podcast_id: str | None = None
    for part in parsed.path.rstrip("/").split("/"):
        if part.startswith("id") and part[2:].isdigit():
            podcast_id = part[2:]
            break
    if not podcast_id:
        raise ValueError(f"Cannot extract podcast ID from Apple Podcasts URL: {url}")

    qs = parse_qs(parsed.query)
    episode_ids = qs.get("i", [])
    episode_id = episode_ids[0] if episode_ids else None
    return podcast_id, episode_id


def _parse_duration(raw: str | int | None) -> int:
    """Parse itunes:duration to seconds.

    Handles: plain integer, "SS", "MM:SS", "HH:MM:SS".
    Returns 0 on any parse failure.
    """
    if raw is None:
        return 0
    s = str(raw).strip()
    if not s:
        return 0
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return int(s)
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return 0


def _struct_to_datetime(t: time.struct_time | None) -> datetime | None:
    """Convert a time.struct_time (from feedparser) to a UTC-aware datetime."""
    if t is None:
        return None
    try:
        return datetime(*t[:6], tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def _thumbnail_from_entry(entry: object, feed: object) -> str | None:
    """Extract thumbnail URL from a feedparser entry or feed object."""
    for obj in (entry, feed):
        img = getattr(obj, "image", None)
        if isinstance(img, dict):
            href = img.get("href") or img.get("url")
            if href:
                return str(href)
    return None


def _best_mp3_entry(
    feed: feedparser.FeedParserDict,
    episode_id: str | None,
) -> tuple[str, feedparser.FeedParserDict]:
    """Find the best episode entry and its MP3 enclosure URL.

    If episode_id is provided, tries to match against iTunes episode IDs in the
    feed (some RSS feeds include itunes:episode or guid). Falls back to latest.
    Returns (mp3_url, entry).
    """
    entries_with_mp3: list[tuple[str, feedparser.FeedParserDict]] = []

    for entry in feed.entries:
        for enc in entry.get("enclosures", []):
            enc_url: str = enc.get("href") or enc.get("url") or ""
            enc_type: str = enc.get("type") or ""
            if "audio" in enc_type or enc_url.lower().endswith(".mp3"):
                entries_with_mp3.append((enc_url, entry))
                break

    if not entries_with_mp3:
        raise ValueError("No episodes with audio enclosures found in the RSS feed.")

    # If we have an episode_id, try matching it against the entry's itunes_episode
    # or the guid (some feeds embed the iTunes trackId in the guid)
    if episode_id:
        for mp3_url, entry in entries_with_mp3:
            guid = str(entry.get("id", ""))
            if episode_id in guid:
                return mp3_url, entry

    # Default: latest episode (first in feed — feeds are newest-first)
    return entries_with_mp3[0]


def _episode_source_item_id(entry: object, mp3_url: str) -> str:
    """Return the feed's stable episode identifier, falling back to its enclosure."""
    if isinstance(entry, dict):
        return str(entry.get("id") or mp3_url)
    return str(getattr(entry, "id", "") or mp3_url)


# ---------------------------------------------------------------------------
# Async helpers — network I/O
# ---------------------------------------------------------------------------

async def _fetch_rss(rss_url: str) -> feedparser.FeedParserDict:
    """Fetch and parse an RSS feed. Runs feedparser in an executor (it's sync)."""
    loop = asyncio.get_event_loop()
    feed: feedparser.FeedParserDict = await loop.run_in_executor(
        None, feedparser.parse, rss_url
    )
    if feed.get("bozo") and not feed.entries:
        raise ValueError(f"Failed to parse RSS feed from {rss_url}: {feed.get('bozo_exception')}")
    return feed


async def _itunes_feed_url(podcast_id: str) -> str:
    """Resolve an Apple Podcasts podcast_id to its RSS feed URL via iTunes Lookup."""
    params = {"id": podcast_id, "entity": "podcast"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(_ITUNES_LOOKUP, params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not results:
        raise ValueError(f"No podcast found for iTunes ID {podcast_id}.")

    feed_url: str | None = results[0].get("feedUrl")
    if not feed_url:
        raise ValueError(f"iTunes returned no RSS feed URL for podcast ID {podcast_id}.")
    return feed_url


async def _download_mp3(url: str, dest: Path, job_id: str = "-") -> None:
    """Stream-download an MP3 file to disk."""
    log = f"job_id={job_id} url={url[:60]!r} source=podcast"
    logger.info("%s event=mp3_download_start", log)
    async with httpx.AsyncClient(follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(_DOWNLOAD_CHUNK):
                    f.write(chunk)
    size_mb = dest.stat().st_size / 1e6
    logger.info("%s event=mp3_download_done size_mb=%.1f", log, size_mb)


# ---------------------------------------------------------------------------
# PodcastSource
# ---------------------------------------------------------------------------

class PodcastSource(BaseSource):
    """Resolves podcast URLs to MP3, transcribes via Whisper, returns TranscriptResult."""

    async def fetch(self, url: str, job_id: str = "-") -> TranscriptResult:
        log = f"job_id={job_id} url={url[:60]!r} source=podcast"
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        try:
            if parsed.path.lower().endswith(".mp3"):
                return await self._from_direct_mp3(url, job_id)

            if hostname == "podcasts.apple.com":
                return await self._from_apple_podcasts(url, job_id)

            # Assume it's an RSS feed URL
            logger.info("%s event=rss_fetch_start", log)
            feed = await _fetch_rss(url)
            return await self._from_feed(feed, url, job_id, episode_id=None)

        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Couldn't process podcast URL: {e}"
            ) from e

    async def _from_direct_mp3(self, url: str, job_id: str) -> TranscriptResult:
        """Download a direct MP3 URL and transcribe it."""
        dest = tmp_path_for_job(job_id)
        await _download_mp3(url, dest, job_id)
        transcript = await transcribe(dest, job_id)

        return TranscriptResult(
            title=Path(urlparse(url).path).stem or "Podcast Episode",
            source="podcast",
            url=url,
            channel_or_show="",
            duration_seconds=0,
            transcript=transcript,
            transcription_model="openai/whisper-1",
            source_item_id=url,
        )

    async def _from_apple_podcasts(self, url: str, job_id: str) -> TranscriptResult:
        """Resolve Apple Podcasts URL → iTunes → RSS → MP3 → transcribe."""
        log = f"job_id={job_id} url={url[:60]!r} source=podcast"
        podcast_id, episode_id = _parse_apple_podcast_ids(url)
        logger.info("%s event=itunes_lookup_start podcast_id=%s", log, podcast_id)
        feed_url = await _itunes_feed_url(podcast_id)
        logger.info("%s event=itunes_lookup_done feed_url=%s", log, feed_url)
        feed = await _fetch_rss(feed_url)
        return await self._from_feed(feed, url, job_id, episode_id=episode_id)

    async def _from_feed(
        self,
        feed: feedparser.FeedParserDict,
        original_url: str,
        job_id: str,
        episode_id: str | None,
    ) -> TranscriptResult:
        """Pick the best episode from a parsed RSS feed, download and transcribe."""
        log = f"job_id={job_id} url={original_url[:60]!r} source=podcast"

        mp3_url, entry = _best_mp3_entry(feed, episode_id)
        source_item_id = _episode_source_item_id(entry, mp3_url)

        title = str(entry.get("title") or "Unknown Episode")
        show_name = str(feed.feed.get("title") or "")
        duration_seconds = _parse_duration(entry.get("itunes_duration"))
        published_at = _struct_to_datetime(entry.get("published_parsed"))
        thumbnail_url = _thumbnail_from_entry(entry, feed.feed)

        logger.info("%s event=episode_selected title=%r", log, title)

        dest = tmp_path_for_job(job_id)
        await _download_mp3(mp3_url, dest, job_id)
        transcript = await transcribe(dest, job_id)

        return TranscriptResult(
            title=title,
            source="podcast",
            url=original_url,
            channel_or_show=show_name,
            duration_seconds=duration_seconds,
            thumbnail_url=thumbnail_url,
            transcript=transcript,
            published_at=published_at,
            transcription_model="openai/whisper-1",
            source_item_id=source_item_id,
        )

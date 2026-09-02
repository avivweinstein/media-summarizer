"""Podcast source implementation.

Supported URL types:
  - Direct MP3 URL (.mp3 extension)
  - RSS feed URL (parsed by feedparser)
  - Apple Podcasts URL (exact episode page, or show lookup → RSS)

Spotify URLs are explicitly rejected with a helpful error message.

Flow for each type:
  1. Resolve URL to (mp3_url, metadata_dict)
  2. Download MP3 to /tmp/media-summarizer/{job_id}.mp3
  3. Transcribe via Whisper (see transcriber.py)
  4. Return TranscriptResult
"""

import asyncio
import html
import ipaddress
import json
import logging
import re
import socket
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import feedparser
import httpx

from config import settings
from exceptions import TranscriptionError, UnsupportedURLError, UsageLimitError
from models import TranscriptResult, UsageStats
from sources.base import BaseSource
from transcriber import tmp_path_for_job, transcribe, transcription_model_name

logger = logging.getLogger(__name__)

_ITUNES_LOOKUP = "https://itunes.apple.com/lookup"
_DOWNLOAD_TIMEOUT = 300  # seconds — large podcasts can be slow
_DOWNLOAD_CHUNK = 64 * 1024  # 64 KB chunks
_RSS_TIMEOUT = 30
_RSS_MAX_BYTES = 5 * 1024 * 1024
_MAX_REDIRECTS = 5


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

    If episode_id is provided, it must match an episode GUID. Show-level URLs
    without an episode ID select the latest episode.
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
        raise ValueError(f"Episode ID {episode_id} was not found in the podcast feed.")

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


async def _validate_public_http_url(url: str) -> str:
    """Return a validated public address to pin for the outbound connection."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsupportedURLError("Media URLs must use HTTP or HTTPS.")
    if parsed.username or parsed.password:
        raise UnsupportedURLError("Media URLs cannot contain credentials.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        answers = await asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as error:
        raise TranscriptionError(f"Could not resolve media host {parsed.hostname}.") from error

    addresses = {str(answer[4][0]) for answer in answers}
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise UnsupportedURLError(
            "Private, loopback, link-local, and reserved media URLs are not allowed."
        )
    return min(
        addresses,
        key=lambda address: ipaddress.ip_address(address).version,
    )


async def _pinned_request_args(
    url: str,
) -> tuple[str, dict[str, str], dict[str, object]]:
    """Build request arguments that connect to the validated address directly."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    address = await _validate_public_http_url(url)
    bracketed_address = f"[{address}]" if ":" in address else address
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port
    pinned_netloc = bracketed_address if port == default_port else f"{bracketed_address}:{port}"
    host_header = hostname if port == default_port else f"{hostname}:{port}"
    pinned_url = urlunparse(
        (
            parsed.scheme,
            pinned_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )
    return pinned_url, {"Host": host_header}, {"sni_hostname": hostname}


def _redirect_target(response: object, current_url: str) -> str | None:
    status_code = getattr(response, "status_code", 200)
    if not isinstance(status_code, int) or not 300 <= status_code < 400:
        return None
    headers = getattr(response, "headers", {})
    location = headers.get("location", "")
    if not location:
        raise UnsupportedURLError("Media redirect did not include a destination.")
    return urljoin(current_url, str(location))


async def _fetch_rss(rss_url: str) -> feedparser.FeedParserDict:
    """Fetch a bounded RSS response and parse it without feedparser network I/O."""
    body = bytearray()
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=_RSS_TIMEOUT,
        trust_env=False,
    ) as client:
        current_url = rss_url
        for _ in range(_MAX_REDIRECTS + 1):
            request_url, headers, extensions = await _pinned_request_args(current_url)
            async with client.stream(
                "GET",
                request_url,
                headers=headers,
                extensions=extensions,
            ) as response:
                redirect = _redirect_target(response, current_url)
                if redirect is not None:
                    current_url = redirect
                    continue
                response.raise_for_status()
                raw_length = response.headers.get("content-length", "")
                if raw_length.isdigit() and int(raw_length) > _RSS_MAX_BYTES:
                    raise UsageLimitError("RSS feed exceeds the configured response-size limit.")
                async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK):
                    body.extend(chunk)
                    if len(body) > _RSS_MAX_BYTES:
                        raise UsageLimitError(
                            "RSS feed exceeds the configured response-size limit."
                        )
                break
        else:
            raise UnsupportedURLError("RSS feed exceeded the redirect limit.")

    feed: feedparser.FeedParserDict = await asyncio.to_thread(
        feedparser.parse,
        bytes(body),
    )
    if feed.get("bozo") and not feed.entries:
        raise UnsupportedURLError(f"URL is not a readable RSS feed: {feed.get('bozo_exception')}")
    if not feed.entries:
        raise UnsupportedURLError("URL is not a podcast RSS feed.")
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


async def _apple_episode_metadata(
    episode_url: str,
    episode_id: str,
) -> dict[str, object]:
    """Resolve an Apple episode page to its exact embedded audio and metadata."""
    body = bytearray()
    async with httpx.AsyncClient(
        timeout=15,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        current_url = episode_url
        for _ in range(_MAX_REDIRECTS + 1):
            if urlparse(current_url).hostname != "podcasts.apple.com":
                raise UnsupportedURLError("Apple Podcasts redirected to an unexpected host.")
            request_url, headers, extensions = await _pinned_request_args(current_url)
            async with client.stream(
                "GET",
                request_url,
                headers=headers,
                extensions=extensions,
            ) as response:
                redirect = _redirect_target(response, current_url)
                if redirect is not None:
                    current_url = redirect
                    continue
                response.raise_for_status()
                async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK):
                    body.extend(chunk)
                    if len(body) > _RSS_MAX_BYTES:
                        raise UsageLimitError(
                            "Apple Podcasts page exceeds the response-size limit."
                        )
                break
        else:
            raise UnsupportedURLError("Apple Podcasts exceeded the redirect limit.")

    page = bytes(body).decode("utf-8", errors="replace")
    content_id = re.search(
        r'<meta\s+name="apple:content_id"\s+content="([^"]+)"',
        page,
    )
    if content_id is None or content_id.group(1) != episode_id:
        raise UnsupportedURLError(f"Apple Podcasts did not resolve episode track ID {episode_id}.")

    metadata: dict[str, object] = {}
    for raw_json in re.findall(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
        page,
        flags=re.DOTALL,
    ):
        try:
            candidate = json.loads(html.unescape(raw_json))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("@type") == "PodcastEpisode":
            metadata = candidate
            break

    audio_match = re.search(r'https?://[^"\'<>\\]+?\.mp3[^"\'<>\\]*', page)
    if not metadata or audio_match is None:
        raise UnsupportedURLError(
            f"Apple Podcasts episode {episode_id} did not expose usable metadata and audio."
        )

    series = metadata.get("partOfSeries")
    series_name = series.get("name", "") if isinstance(series, dict) else ""
    duration_seconds = _parse_iso_duration(str(metadata.get("duration") or ""))
    return {
        "trackId": episode_id,
        "episodeUrl": html.unescape(audio_match.group(0)),
        "trackName": metadata.get("name", "Unknown Episode"),
        "collectionName": series_name,
        "trackTimeMillis": duration_seconds * 1000,
        "releaseDate": metadata.get("datePublished"),
        "artworkUrl600": metadata.get("thumbnailUrl"),
    }


def _parse_iso_duration(raw: str) -> int:
    match = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        raw,
    )
    if match is None:
        return 0
    hours, minutes, seconds = (int(value or 0) for value in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _parse_iso_datetime(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


async def _download_mp3(url: str, dest: Path, job_id: str = "-") -> None:
    """Stream-download an MP3 file to disk."""
    log = f"job_id={job_id} url={url[:60]!r} source=podcast"
    logger.info("%s event=mp3_download_start", log)
    downloaded = 0
    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=_DOWNLOAD_TIMEOUT,
            trust_env=False,
        ) as client:
            current_url = url
            for _ in range(_MAX_REDIRECTS + 1):
                request_url, headers, extensions = await _pinned_request_args(current_url)
                async with client.stream(
                    "GET",
                    request_url,
                    headers=headers,
                    extensions=extensions,
                ) as resp:
                    redirect = _redirect_target(resp, current_url)
                    if redirect is not None:
                        current_url = redirect
                        continue
                    resp.raise_for_status()
                    raw_content_length = resp.headers.get("content-length", "")
                    content_length = int(raw_content_length) if raw_content_length.isdigit() else 0
                    if content_length > settings.max_audio_download_bytes:
                        raise UsageLimitError(
                            "Podcast audio exceeds the configured download-size limit."
                        )
                    with open(dest, "wb") as f:
                        async for chunk in resp.aiter_bytes(_DOWNLOAD_CHUNK):
                            downloaded += len(chunk)
                            if downloaded > settings.max_audio_download_bytes:
                                raise UsageLimitError(
                                    "Podcast audio exceeds the configured download-size limit."
                                )
                            f.write(chunk)
                    break
            else:
                raise UnsupportedURLError("Podcast audio exceeded the redirect limit.")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    size_mb = dest.stat().st_size / 1e6
    logger.info("%s event=mp3_download_done size_mb=%.1f", log, size_mb)


# ---------------------------------------------------------------------------
# PodcastSource
# ---------------------------------------------------------------------------


class PodcastSource(BaseSource):
    """Resolves podcast URLs to MP3, transcribes via Whisper, returns TranscriptResult."""

    async def fetch(
        self,
        url: str,
        job_id: str = "-",
        *,
        usage: UsageStats | None = None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None = None,
        processing_mode: str = "nvidia_internal",
    ) -> TranscriptResult:
        log = f"job_id={job_id} url={url[:60]!r} source=podcast"
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        try:
            if parsed.path.lower().endswith(".mp3"):
                return await self._from_direct_mp3(
                    url, job_id, usage, persist_usage, processing_mode
                )

            if hostname == "podcasts.apple.com":
                return await self._from_apple_podcasts(
                    url, job_id, usage, persist_usage, processing_mode
                )

            # Assume it's an RSS feed URL
            logger.info("%s event=rss_fetch_start", log)
            feed = await _fetch_rss(url)
            return await self._from_feed(
                feed,
                url,
                job_id,
                episode_id=None,
                usage=usage,
                persist_usage=persist_usage,
                processing_mode=processing_mode,
            )

        except (TranscriptionError, UnsupportedURLError, UsageLimitError):
            raise
        except Exception as e:
            raise TranscriptionError(f"Couldn't process podcast URL: {e}") from e

    async def _from_direct_mp3(
        self,
        url: str,
        job_id: str,
        usage: UsageStats | None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None,
        processing_mode: str = "nvidia_internal",
    ) -> TranscriptResult:
        """Download a direct MP3 URL and transcribe it."""
        dest = tmp_path_for_job(job_id)
        await _download_mp3(url, dest, job_id)
        transcription = await transcribe(
            dest,
            job_id,
            usage=usage,
            persist_usage=persist_usage,
            processing_mode=processing_mode,
        )

        return TranscriptResult(
            title=Path(urlparse(url).path).stem or "Podcast Episode",
            source="podcast",
            url=url,
            channel_or_show="",
            duration_seconds=0,
            transcript=transcription.text,
            segments=transcription.segments,
            transcription_model=transcription_model_name(processing_mode),
            source_item_id=url,
        )

    async def _from_apple_podcasts(
        self,
        url: str,
        job_id: str,
        usage: UsageStats | None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None,
        processing_mode: str = "nvidia_internal",
    ) -> TranscriptResult:
        """Resolve Apple Podcasts URL → iTunes → RSS → MP3 → transcribe."""
        log = f"job_id={job_id} url={url[:60]!r} source=podcast"
        podcast_id, episode_id = _parse_apple_podcast_ids(url)
        if episode_id:
            logger.info("%s event=itunes_episode_lookup_start episode_id=%s", log, episode_id)
            episode = await _apple_episode_metadata(url, episode_id)
            audio_url = str(episode["episodeUrl"])
            duration_seconds = int(str(episode.get("trackTimeMillis") or 0)) // 1000
            dest = tmp_path_for_job(job_id)
            await _download_mp3(audio_url, dest, job_id)
            transcription = await transcribe(
                dest,
                job_id,
                duration_seconds=duration_seconds,
                usage=usage,
                persist_usage=persist_usage,
                processing_mode=processing_mode,
            )
            return TranscriptResult(
                title=str(episode.get("trackName") or "Unknown Episode"),
                source="podcast",
                url=url,
                channel_or_show=str(episode.get("collectionName") or ""),
                duration_seconds=duration_seconds,
                thumbnail_url=(
                    str(episode["artworkUrl600"]) if episode.get("artworkUrl600") else None
                ),
                transcript=transcription.text,
                segments=transcription.segments,
                published_at=_parse_iso_datetime(episode.get("releaseDate")),
                transcription_model=transcription_model_name(processing_mode),
                source_item_id=episode_id,
            )

        logger.info("%s event=itunes_lookup_start podcast_id=%s", log, podcast_id)
        feed_url = await _itunes_feed_url(podcast_id)
        logger.info("%s event=itunes_lookup_done feed_url=%s", log, feed_url)
        feed = await _fetch_rss(feed_url)
        return await self._from_feed(
            feed,
            url,
            job_id,
            episode_id=episode_id,
            usage=usage,
            persist_usage=persist_usage,
            processing_mode=processing_mode,
        )

    async def _from_feed(
        self,
        feed: feedparser.FeedParserDict,
        original_url: str,
        job_id: str,
        episode_id: str | None,
        usage: UsageStats | None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None,
        processing_mode: str = "nvidia_internal",
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
        transcription = await transcribe(
            dest,
            job_id,
            duration_seconds=duration_seconds,
            usage=usage,
            persist_usage=persist_usage,
            processing_mode=processing_mode,
        )

        return TranscriptResult(
            title=title,
            source="podcast",
            url=original_url,
            channel_or_show=show_name,
            duration_seconds=duration_seconds,
            thumbnail_url=thumbnail_url,
            transcript=transcription.text,
            segments=transcription.segments,
            published_at=published_at,
            transcription_model=transcription_model_name(processing_mode),
            source_item_id=source_item_id,
        )

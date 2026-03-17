"""YouTube source — transcript via youtube-transcript-api, metadata via yt-dlp.

No transcript → NoTranscriptError. No audio fallback, ever.
"""

import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from exceptions import MetadataError, NoTranscriptError
from models import TranscriptResult
from sources.base import BaseSource

logger = logging.getLogger(__name__)


def _extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname == "youtu.be":
        vid = parsed.path.lstrip("/").split("?")[0]
        if not vid:
            raise ValueError(f"Cannot extract video ID from: {url}")
        return vid
    qs = parse_qs(parsed.query)
    ids = qs.get("v", [])
    if not ids:
        raise ValueError(f"Cannot extract video ID from: {url}")
    return ids[0]


def _fetch_transcript_sync(video_id: str) -> str:
    """Fetch transcript text. Tries English first, falls back to any available language."""
    api = YouTubeTranscriptApi()
    try:
        listing = api.list(video_id)
    except TranscriptsDisabled:
        raise NoTranscriptError("Transcripts are disabled for this video.")
    except VideoUnavailable:
        raise NoTranscriptError("Video is unavailable (private or deleted).")
    except Exception as e:
        raise NoTranscriptError(f"Could not list transcripts: {e}") from e

    available = list(listing)
    if not available:
        raise NoTranscriptError("No transcript available for this video. Skipping.")

    # Prefer human-made English, then any English, then first available
    try:
        transcript_obj = listing.find_transcript(["en", "en-US", "en-GB", "en-CA", "en-AU"])
    except NoTranscriptFound:
        transcript_obj = available[0]
        logger.warning(
            "url=- source=youtube event=non_english_transcript lang=%s",
            transcript_obj.language_code,
        )

    try:
        fetched = transcript_obj.fetch()
    except Exception as e:
        raise NoTranscriptError(f"Failed to fetch transcript content: {e}") from e

    return " ".join(snippet.text for snippet in fetched)


def _fetch_metadata_sync(url: str, youtube_api_key: str = "") -> dict:  # type: ignore[type-arg]
    """Fetch video metadata via yt-dlp without downloading any media."""
    opts: dict = {  # type: ignore[type-arg]
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    if youtube_api_key:
        opts["youtube_api_key"] = youtube_api_key

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            raise MetadataError(f"Failed to fetch YouTube metadata: {e}") from e

    if info is None:
        raise MetadataError("yt-dlp returned no metadata for this URL.")
    return info  # type: ignore[return-value]


def _parse_upload_date(raw: str | None) -> datetime | None:
    """Parse yt-dlp upload_date string 'YYYYMMDD' into a UTC datetime."""
    if not raw or len(raw) != 8:
        return None
    try:
        return datetime(int(raw[:4]), int(raw[4:6]), int(raw[6:]), tzinfo=timezone.utc)
    except ValueError:
        return None


class YouTubeSource(BaseSource):
    def __init__(self, youtube_api_key: str = "") -> None:
        self.youtube_api_key = youtube_api_key

    async def fetch(self, url: str, job_id: str = "-") -> TranscriptResult:
        log = f"job_id={job_id} url={url[:60]!r} source=youtube"
        loop = asyncio.get_event_loop()

        logger.info("%s event=transcript_fetch_start", log)
        transcript = await loop.run_in_executor(
            None, _fetch_transcript_sync, _extract_video_id(url)
        )
        logger.info("%s event=transcript_fetch_done chars=%d", log, len(transcript))

        logger.info("%s event=metadata_fetch_start", log)
        meta = await loop.run_in_executor(
            None, _fetch_metadata_sync, url, self.youtube_api_key
        )
        logger.info("%s event=metadata_fetch_done title=%r", log, meta.get("title"))

        return TranscriptResult(
            title=meta.get("title") or "Unknown Title",
            source="youtube",
            url=url,
            channel_or_show=meta.get("channel") or meta.get("uploader") or "",
            duration_seconds=int(meta.get("duration") or 0),
            thumbnail_url=meta.get("thumbnail"),
            transcript=transcript,
            published_at=_parse_upload_date(meta.get("upload_date")),
        )

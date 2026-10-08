"""YouTube source — transcript via youtube-transcript-api, metadata via yt-dlp.

If no native transcript is available, falls back to downloading audio via yt-dlp
and transcribing with OpenAI Whisper.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from exceptions import MetadataError
from models import TranscriptResult
from sources.base import BaseSource
from transcriber import tmp_path_for_job, transcribe

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


def _fetch_transcript_sync(video_id: str) -> str | None:
    """Fetch transcript text. Returns None if no transcript is available.

    Tries native English transcript first, falls back to any language.
    Returns None (instead of raising) so the caller can fall back to Whisper.
    """
    api = YouTubeTranscriptApi()
    try:
        listing = api.list(video_id)
    except (TranscriptsDisabled, VideoUnavailable):
        return None
    except Exception:
        return None

    available = list(listing)
    if not available:
        return None

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
    except Exception:
        return None

    return " ".join(snippet.text for snippet in fetched)


def _fetch_metadata_sync(url: str, youtube_api_key: str = "") -> dict[str, object]:
    """Fetch video metadata via yt-dlp without downloading any media."""
    opts: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    if youtube_api_key:
        opts["youtube_api_key"] = youtube_api_key

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info: dict[str, object] | None = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            raise MetadataError(f"Failed to fetch YouTube metadata: {e}") from e

    if info is None:
        raise MetadataError("yt-dlp returned no metadata for this URL.")
    return info


def _parse_upload_date(raw: str | None) -> datetime | None:
    """Parse yt-dlp upload_date string 'YYYYMMDD' into a UTC datetime."""
    if not raw or len(raw) != 8:
        return None
    try:
        return datetime(int(raw[:4]), int(raw[4:6]), int(raw[6:]), tzinfo=UTC)
    except ValueError:
        return None


def _download_audio_sync(url: str, dest: Path) -> None:
    """Download audio from a YouTube video via yt-dlp as MP3."""
    opts: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(dest.with_suffix(".%(ext)s")),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    # yt-dlp may produce dest.mp3 — rename to the exact path we want
    actual = dest.with_suffix(".mp3")
    if actual.exists() and actual != dest:
        actual.rename(dest)


class YouTubeSource(BaseSource):
    def __init__(self, youtube_api_key: str = "") -> None:
        self.youtube_api_key = youtube_api_key

    async def fetch(self, url: str, job_id: str = "-") -> TranscriptResult:
        log = f"job_id={job_id} url={url[:60]!r} source=youtube"
        loop = asyncio.get_event_loop()

        # Try native transcript first
        logger.info("%s event=transcript_fetch_start", log)
        transcript = await loop.run_in_executor(
            None, _fetch_transcript_sync, _extract_video_id(url)
        )

        if transcript:
            logger.info("%s event=transcript_fetch_done chars=%d method=native", log, len(transcript))
        else:
            # Fall back to Whisper: download audio, transcribe
            logger.info("%s event=transcript_native_unavailable fallback=whisper", log)
            dest = tmp_path_for_job(job_id)
            logger.info("%s event=audio_download_start", log)
            await loop.run_in_executor(None, _download_audio_sync, url, dest)
            logger.info(
                "%s event=audio_download_done size_mb=%.1f",
                log, dest.stat().st_size / 1e6 if dest.exists() else 0,
            )
            transcript = await transcribe(dest, job_id)
            logger.info("%s event=transcript_fetch_done chars=%d method=whisper", log, len(transcript))

        logger.info("%s event=metadata_fetch_start", log)
        meta = await loop.run_in_executor(
            None, _fetch_metadata_sync, url, self.youtube_api_key
        )
        logger.info("%s event=metadata_fetch_done title=%r", log, meta.get("title"))

        raw_thumb = meta.get("thumbnail")
        thumbnail_url = str(raw_thumb) if raw_thumb else None
        raw_date = meta.get("upload_date")
        upload_date = str(raw_date) if raw_date else None

        return TranscriptResult(
            title=str(meta.get("title") or "Unknown Title"),
            source="youtube",
            url=url,
            channel_or_show=str(meta.get("channel") or meta.get("uploader") or ""),
            duration_seconds=int(str(meta.get("duration") or 0)),
            thumbnail_url=thumbnail_url,
            transcript=transcript,
            published_at=_parse_upload_date(upload_date),
        )

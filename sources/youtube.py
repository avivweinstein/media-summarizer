"""YouTube source — transcript via youtube-transcript-api, metadata via yt-dlp.

If no native transcript is available, falls back to downloading audio via yt-dlp
and transcribing with OpenAI Whisper.
"""

import logging
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from async_utils import run_blocking
from config import settings
from exceptions import MetadataError, UsageLimitError
from models import TranscriptionOutput, TranscriptResult, TranscriptSegment, UsageStats
from sources.base import BaseSource
from transcriber import tmp_path_for_job, transcribe, transcription_model_name

logger = logging.getLogger(__name__)


class _TimeoutSession(requests.Session):
    def request(
        self, method: str | bytes, url: str | bytes, *args: Any, **kwargs: Any
    ) -> requests.Response:
        kwargs.setdefault("timeout", settings.source_fetch_timeout_seconds)
        return super().request(method, url, *args, **kwargs)


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


def _fetch_transcript_sync(video_id: str) -> TranscriptionOutput | None:
    """Fetch transcript text and timestamps, or None when unavailable.

    Tries native English transcript first, falls back to any language.
    Returns None (instead of raising) so the caller can fall back to Whisper.
    """
    with _TimeoutSession() as session:
        api = YouTubeTranscriptApi(http_client=session)
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
            transcript_obj = listing.find_transcript(
                ["en", "en-US", "en-GB", "en-CA", "en-AU"]
            )
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

    segments = [
        TranscriptSegment(
            start_seconds=snippet.start,
            end_seconds=snippet.start + snippet.duration,
            text=snippet.text,
        )
        for snippet in fetched
        if snippet.text.strip()
    ]
    return TranscriptionOutput(
        text=" ".join(segment.text for segment in segments),
        segments=segments,
    )


def _fetch_metadata_sync(
    url: str, youtube_api_key: str = "", no_playlist: bool = False
) -> dict[str, object]:
    """Fetch video metadata via yt-dlp without downloading any media."""
    opts: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "proxy": "",
        "skip_download": True,
        "socket_timeout": settings.source_fetch_timeout_seconds,
    }
    if youtube_api_key:
        opts["youtube_api_key"] = youtube_api_key
    if no_playlist:
        opts["noplaylist"] = True

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info: dict[str, object] | None = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            raise MetadataError(f"Failed to fetch media metadata: {e}") from e

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


def _download_audio_sync(
    url: str,
    dest: Path,
    max_bytes: int,
    cancel_event: threading.Event | None = None,
    no_playlist: bool = False,
    media_info: dict[str, object] | None = None,
) -> None:
    """Download audio from one hosted video via yt-dlp as MP3."""

    def enforce_size(progress: dict[str, object]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Media download cancelled.")
        downloaded = progress.get("downloaded_bytes", 0)
        if isinstance(downloaded, (int, float)) and downloaded > max_bytes:
            raise UsageLimitError("Media audio exceeds the configured download-size limit.")

    opts: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "proxy": "",
        "format": "bestaudio/best",
        "max_filesize": max_bytes,
        "outtmpl": str(dest.with_suffix(".%(ext)s")),
        "progress_hooks": [enforce_size],
        "socket_timeout": settings.source_fetch_timeout_seconds,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
    }
    if no_playlist:
        opts["noplaylist"] = True
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            if media_info is not None:
                ydl.process_info(dict(media_info))
                result = 0
            else:
                result = ydl.download([url])
        if result != 0:
            raise MetadataError("yt-dlp could not download the media audio.")
        # yt-dlp may produce dest.mp3 — rename to the exact path we want
        actual = dest.with_suffix(".mp3")
        if actual.exists() and actual != dest:
            actual.rename(dest)
        if not dest.exists():
            raise UsageLimitError(
                "YouTube audio was not downloaded; it may exceed the configured size limit."
            )
    except yt_dlp.utils.DownloadError as error:
        for partial in dest.parent.glob(f"{dest.stem}.*"):
            partial.unlink(missing_ok=True)
        if "download-size limit" in str(error):
            raise UsageLimitError(
                "Media audio exceeds the configured download-size limit."
            ) from error
        raise
    except Exception:
        for partial in dest.parent.glob(f"{dest.stem}.*"):
            partial.unlink(missing_ok=True)
        raise


class YouTubeSource(BaseSource):
    def __init__(self, youtube_api_key: str = "") -> None:
        self.youtube_api_key = youtube_api_key

    async def fetch(
        self,
        url: str,
        job_id: str = "-",
        *,
        usage: UsageStats | None = None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None = None,
        processing_mode: str = "cloud_public",
    ) -> TranscriptResult:
        log = f"job_id={job_id} url={url[:60]!r} source=youtube"
        video_id = _extract_video_id(url)

        logger.info("%s event=metadata_fetch_start", log)
        meta = await run_blocking(_fetch_metadata_sync, url, self.youtube_api_key)
        logger.info("%s event=metadata_fetch_done title=%r", log, meta.get("title"))
        duration_seconds = int(str(meta.get("duration") or 0))
        usage_tracker = usage or UsageStats()

        # Try native transcript first
        logger.info("%s event=transcript_fetch_start", log)
        transcription = await run_blocking(_fetch_transcript_sync, video_id)

        if transcription and transcription.text:
            transcription_model = "youtube/captions"
            logger.info(
                "%s event=transcript_fetch_done chars=%d method=native",
                log,
                len(transcription.text),
            )
        else:
            # Fall back to Whisper: download audio, transcribe
            logger.info("%s event=transcript_native_unavailable fallback=whisper", log)
            if duration_seconds > settings.max_audio_duration_seconds:
                raise UsageLimitError(
                    f"Audio duration is {duration_seconds / 3600:.1f} hours, exceeding the "
                    f"configured {settings.max_audio_duration_seconds / 3600:.1f}-hour limit."
                )
            dest = tmp_path_for_job(job_id)
            logger.info("%s event=audio_download_start", log)
            cancel_event = threading.Event()
            await run_blocking(
                _download_audio_sync,
                url,
                dest,
                settings.max_audio_download_bytes,
                cancel_event,
                cancel=cancel_event.set,
            )
            logger.info(
                "%s event=audio_download_done size_mb=%.1f",
                log,
                dest.stat().st_size / 1e6 if dest.exists() else 0,
            )
            transcription = await transcribe(
                dest,
                job_id,
                duration_seconds=duration_seconds,
                usage=usage_tracker,
                persist_usage=persist_usage,
                processing_mode=processing_mode,
            )
            transcription_model = transcription_model_name(processing_mode)
            logger.info(
                "%s event=transcript_fetch_done chars=%d method=whisper",
                log,
                len(transcription.text),
            )

        raw_thumb = meta.get("thumbnail")
        thumbnail_url = str(raw_thumb) if raw_thumb else None
        raw_date = meta.get("upload_date")
        upload_date = str(raw_date) if raw_date else None

        return TranscriptResult(
            title=str(meta.get("title") or "Unknown Title"),
            source="youtube",
            url=url,
            channel_or_show=str(meta.get("channel") or meta.get("uploader") or ""),
            duration_seconds=duration_seconds,
            thumbnail_url=thumbnail_url,
            transcript=transcription.text,
            segments=transcription.segments,
            published_at=_parse_upload_date(upload_date),
            transcription_model=transcription_model,
            source_item_id=video_id,
        )

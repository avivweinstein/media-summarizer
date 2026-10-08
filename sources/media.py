"""Generic hosted media source, including Vimeo and X/Twitter."""

import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlparse, urlunparse

import yt_dlp

from async_utils import run_blocking
from config import settings
from exceptions import MetadataError, UnsupportedURLError, UsageLimitError
from models import TranscriptResult, UsageStats
from sources.podcast import _download_mp3, _validate_public_http_url
from sources.youtube import (
    _download_audio_sync,
    _fetch_metadata_sync,
    _HostRestrictedYoutubeDL,
)
from transcriber import convert_to_mp3, tmp_path_for_job, transcribe, transcription_model_name
from url_identity import twitter_status_parts

_TWITTER_MEDIA_HOST = "video.twimg.com"
_TWITTER_REQUEST_HOSTS = frozenset(
    {
        "amp.twimg.com",
        "api.twitter.com",
        "api.x.com",
        "cdn.syndication.twimg.com",
        "m.twitter.com",
        "m.x.com",
        "mobile.twitter.com",
        "mobile.x.com",
        "pbs.twimg.com",
        "twitter.com",
        "video.twimg.com",
        "www.twitter.com",
        "www.x.com",
        "x.com",
    }
)


def _vimeo_player_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "player.vimeo.com":
        return url
    video_ids = [part for part in parsed.path.split("/") if part.isdigit()]
    if not video_ids:
        raise UnsupportedURLError("Could not identify the Vimeo video ID.")
    return f"https://player.vimeo.com/video/{video_ids[-1]}"


def _twitter_base_url(url: str) -> str:
    status = twitter_status_parts(url)
    if status is None:
        raise MetadataError("Invalid X/Twitter post URL.")
    status_id, _ = status
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    status_position = next(
        index for index, part in enumerate(parts) if part in {"status", "statuses"}
    )
    path = "/" + "/".join([*parts[: status_position + 1], status_id])
    return urlunparse(parsed._replace(path=path, query="", fragment=""))


def _fetch_twitter_metadata_sync(url: str) -> dict[str, object]:
    """Extract only native X media without following link cards or external URLs."""
    options: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "proxy": "",
        "skip_download": True,
        "format": "bestaudio/best",
        "socket_timeout": settings.source_fetch_timeout_seconds,
    }
    base_url = _twitter_base_url(url)
    try:
        with _HostRestrictedYoutubeDL(options, _TWITTER_REQUEST_HOSTS) as ydl:
            raw = ydl.extract_info(base_url, download=False, process=False)
    except yt_dlp.utils.DownloadError as error:
        raise MetadataError(
            "The X post has no downloadable video, is unavailable, or requires login."
        ) from error
    if not isinstance(raw, dict) or raw.get("_type") in {"url", "url_transparent"}:
        raise MetadataError("The X post does not contain native video.")

    if raw.get("_type") == "playlist":
        entries = [entry for entry in raw.get("entries") or [] if isinstance(entry, dict)]
        if len(entries) > 1:
            raise MetadataError(
                "X posts containing multiple videos are not supported yet."
            )
        if not entries:
            raise MetadataError("The X post does not contain downloadable video.")
        raw = cast(dict[str, object], entries[0])

    if raw.get("_type") not in {None, "video"}:
        raise MetadataError("The X post does not contain native video.")
    formats = [item for item in raw.get("formats") or [] if isinstance(item, dict)]
    if not formats:
        raise MetadataError("The X post does not contain downloadable video.")
    for item in formats:
        media_url = item.get("url")
        if not isinstance(media_url, str):
            raise MetadataError("The X post returned invalid media metadata.")
        parsed = urlparse(media_url)
        if parsed.scheme != "https" or parsed.hostname != _TWITTER_MEDIA_HOST:
            raise MetadataError("The X post delegated to an untrusted media host.")
    raw["extractor"] = "twitter"
    raw["extractor_key"] = "Twitter"
    with _HostRestrictedYoutubeDL(options, _TWITTER_REQUEST_HOSTS) as ydl:
        processed = ydl.process_ie_result(dict(raw), download=False)
    if not isinstance(processed, dict):
        raise MetadataError("The X post returned invalid media metadata.")
    return cast(dict[str, object], processed)


def _published_at(metadata: dict[str, object]) -> datetime | None:
    timestamp = metadata.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(timestamp, UTC)
    except (OSError, OverflowError, ValueError):
        return None


class MediaSource:
    async def fetch(
        self,
        url: str,
        job_id: str = "-",
        *,
        usage: UsageStats | None = None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None = None,
        processing_mode: str = "cloud_public",
    ) -> TranscriptResult:
        hostname = (urlparse(url).hostname or "").casefold().removeprefix("www.")
        is_vimeo = hostname in {"vimeo.com", "player.vimeo.com"}
        is_twitter = twitter_status_parts(url) is not None
        is_hosted_media = is_vimeo or is_twitter
        metadata: dict[str, object] = {}
        destination = tmp_path_for_job(job_id)
        raw_destination: Path | None = None
        if is_hosted_media:
            media_url = _vimeo_player_url(url) if is_vimeo else url
            await _validate_public_http_url(media_url)
            if is_twitter:
                metadata = await run_blocking(_fetch_twitter_metadata_sync, media_url)
            else:
                metadata = await run_blocking(_fetch_metadata_sync, media_url)
            age_limit = metadata.get("age_limit")
            if is_twitter and isinstance(age_limit, (int, float)) and age_limit > 0:
                raise UnsupportedURLError("Age-gated X videos are not supported.")
        else:
            media_url = url
            raw_suffix = Path(urlparse(url).path).suffix.casefold() or ".media"
            raw_destination = destination.with_suffix(raw_suffix)
            await _download_mp3(url, raw_destination, job_id)
            try:
                await convert_to_mp3(raw_destination, destination)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            finally:
                raw_destination.unlink(missing_ok=True)
        duration_seconds = int(str(metadata.get("duration") or 0))
        if duration_seconds > settings.max_audio_duration_seconds:
            raise UsageLimitError("Media exceeds the configured duration limit.")
        if is_hosted_media:
            cancel_event = threading.Event()
            await run_blocking(
                _download_audio_sync,
                media_url,
                destination,
                settings.max_audio_download_bytes,
                cancel_event,
                is_twitter,
                metadata if is_twitter else None,
                _TWITTER_REQUEST_HOSTS if is_twitter else None,
                cancel=cancel_event.set,
            )
        transcription = await transcribe(
            destination,
            job_id,
            duration_seconds=duration_seconds,
            usage=usage,
            persist_usage=persist_usage,
            processing_mode=processing_mode,
        )
        thumbnail = metadata.get("thumbnail")
        source_item_id = metadata.get("id") or metadata.get("webpage_url") or url
        return TranscriptResult(
            title=str(metadata.get("title") or Path(urlparse(url).path).name or "Unknown media"),
            source="twitter" if is_twitter else "media",
            url=url,
            channel_or_show=str(metadata.get("channel") or metadata.get("uploader") or ""),
            duration_seconds=duration_seconds,
            thumbnail_url=str(thumbnail) if thumbnail else None,
            published_at=_published_at(metadata),
            transcript=transcription.text,
            segments=transcription.segments,
            transcription_model=transcription_model_name(processing_mode),
            source_item_id=str(source_item_id),
        )

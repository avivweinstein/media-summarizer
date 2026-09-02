"""Generic hosted media source, including Vimeo."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlparse

from config import settings
from exceptions import UnsupportedURLError, UsageLimitError
from models import TranscriptResult, UsageStats
from sources.podcast import _download_mp3, _validate_public_http_url
from sources.youtube import _download_audio_sync, _fetch_metadata_sync
from transcriber import tmp_path_for_job, transcribe, transcription_model_name


def _vimeo_player_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "player.vimeo.com":
        return url
    video_ids = [part for part in parsed.path.split("/") if part.isdigit()]
    if not video_ids:
        raise UnsupportedURLError("Could not identify the Vimeo video ID.")
    return f"https://player.vimeo.com/video/{video_ids[-1]}"


class MediaSource:
    async def fetch(
        self,
        url: str,
        job_id: str = "-",
        *,
        usage: UsageStats | None = None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None = None,
    ) -> TranscriptResult:
        loop = asyncio.get_running_loop()
        hostname = (urlparse(url).hostname or "").casefold()
        is_vimeo = hostname in {"vimeo.com", "player.vimeo.com"}
        metadata: dict[str, object] = {}
        destination = tmp_path_for_job(job_id)
        if is_vimeo:
            media_url = _vimeo_player_url(url)
            await _validate_public_http_url(media_url)
            metadata = await loop.run_in_executor(None, _fetch_metadata_sync, media_url)
        else:
            media_url = url
            await _download_mp3(url, destination, job_id)
        duration_seconds = int(str(metadata.get("duration") or 0))
        if duration_seconds > settings.max_audio_duration_seconds:
            raise UsageLimitError("Media exceeds the configured duration limit.")
        if is_vimeo:
            await loop.run_in_executor(
                None,
                _download_audio_sync,
                media_url,
                destination,
                settings.max_audio_download_bytes,
            )
        transcription = await transcribe(
            destination,
            job_id,
            duration_seconds=duration_seconds,
            usage=usage,
            persist_usage=persist_usage,
        )
        thumbnail = metadata.get("thumbnail")
        source_item_id = metadata.get("id") or metadata.get("webpage_url") or url
        return TranscriptResult(
            title=str(metadata.get("title") or Path(urlparse(url).path).name or "Unknown media"),
            source="media",
            url=url,
            channel_or_show=str(metadata.get("channel") or metadata.get("uploader") or ""),
            duration_seconds=duration_seconds,
            thumbnail_url=str(thumbnail) if thumbnail else None,
            transcript=transcription.text,
            segments=transcription.segments,
            transcription_model=transcription_model_name(),
            source_item_id=str(source_item_id),
        )

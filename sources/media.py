"""Generic hosted media source, including Vimeo and X/Twitter."""

import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlparse

from async_utils import run_blocking
from config import settings
from exceptions import MetadataError, UnsupportedURLError, UsageLimitError
from models import TranscriptResult, UsageStats
from sources.podcast import _download_mp3, _validate_public_http_url
from sources.youtube import _download_audio_sync, _fetch_metadata_sync
from transcriber import convert_to_mp3, tmp_path_for_job, transcribe, transcription_model_name
from url_identity import twitter_status_parts


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
            try:
                metadata = await run_blocking(
                    _fetch_metadata_sync,
                    media_url,
                    "",
                    is_twitter,
                )
            except MetadataError as error:
                if is_twitter:
                    raise MetadataError(
                        "The X post has no downloadable video, is unavailable, or requires login."
                    ) from error
                raise
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
            transcript=transcription.text,
            segments=transcription.segments,
            transcription_model=transcription_model_name(processing_mode),
            source_item_id=str(source_item_id),
        )

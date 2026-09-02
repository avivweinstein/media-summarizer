"""Pipeline orchestrator.

Coordinates: detect source → fetch → summarize → configured output writers.
Failed jobs are retried up to MAX_RETRIES times with exponential backoff.
On completion (success or final failure), fires the per-job webhook if configured.
Playlist/bulk URLs are expanded into individual jobs.
"""

import asyncio
import logging
from urllib.parse import parse_qs, urlparse

import httpx
import yt_dlp

import job_queue
from config import settings
from exceptions import UnsupportedURLError, UsageLimitError
from models import Job, JobStage, JobStatus, TranscriptResult, UsageStats
from notion_writer import save_to_notion
from obsidian_writer import save_to_obsidian
from sources.article import ArticleSource
from sources.media import MediaSource
from sources.podcast import PodcastSource
from sources.upload import UploadSource, cleanup_upload
from sources.youtube import YouTubeSource
from summarizer import summarize, summary_model_name

logger = logging.getLogger(__name__)

_YOUTUBE_HOSTNAMES = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}
_SPOTIFY_HOSTNAMES = {"open.spotify.com"}
_APPLE_HOSTNAMES = {"podcasts.apple.com"}
_MEDIA_HOSTNAMES = {"vimeo.com", "player.vimeo.com"}
_MEDIA_EXTENSIONS = {".m4a", ".mov", ".mp4", ".wav", ".webm"}
_UNSUPPORTED_MEDIA_HOSTNAMES = {
    "soundcloud.com",
    "twitter.com",
    "x.com",
}

MAX_RETRIES = 3
_BACKOFF_SECONDS = [5, 10, 20]  # sleep before attempt 2, 3, 4 (never used for attempt 1)


def detect_source(url: str) -> str:
    """Return source type ('youtube' | 'podcast' | 'youtube_playlist') or raise UnsupportedURLError."""
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower().lstrip("www.")

    if parsed.scheme == "upload":
        return "upload"

    if hostname in {"youtube.com", "youtu.be", "m.youtube.com"}:
        qs = parse_qs(parsed.query or "")
        # Playlist URL: has 'list' param but no 'v' param, or is /playlist path
        is_playlist = "playlist" in parsed.path or ("list" in qs and "v" not in qs)
        if is_playlist:
            return "youtube_playlist"

        is_watch = "watch" in parsed.path or parsed.hostname == "youtu.be"
        has_v = "v" in qs
        if not (is_watch or has_v):
            raise UnsupportedURLError("Only YouTube video URLs or playlist URLs are supported.")
        return "youtube"

    if hostname == "open.spotify.com":
        raise UnsupportedURLError(
            "Spotify does not expose a reliable canonical RSS mapping. Paste the show's RSS, "
            "Apple Podcasts, or direct episode URL instead."
        )

    lower_path = parsed.path.lower()
    if hostname in _MEDIA_HOSTNAMES or any(lower_path.endswith(ext) for ext in _MEDIA_EXTENSIONS):
        return "media"

    if hostname == "podcasts.apple.com" or lower_path.endswith(".mp3"):
        return "podcast"

    if parsed.scheme in {"http", "https"} and hostname:
        if hostname in _UNSUPPORTED_MEDIA_HOSTNAMES:
            raise UnsupportedURLError(
                "This media host is not currently supported. Try YouTube, Apple Podcasts, "
                "a podcast RSS URL, or a direct MP3 URL."
            )
        return "article"

    raise UnsupportedURLError(
        "Unsupported source. Supported: YouTube (video or playlist), "
        "Apple Podcasts, RSS feeds, or direct MP3."
    )


def _extract_playlist_videos_sync(url: str) -> list[str]:
    """Extract individual video URLs from a YouTube playlist via yt-dlp."""
    opts: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if info is None:
        return []

    entries = info.get("entries") or []
    urls: list[str] = []
    for entry in entries:
        vid = entry.get("id") or entry.get("url")
        if vid:
            urls.append(f"https://www.youtube.com/watch?v={vid}")
    return urls


async def expand_playlist(url: str) -> list[str]:
    """Expand a playlist URL into individual video URLs. Runs yt-dlp in executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_playlist_videos_sync, url)


async def _notify_webhook(
    job: Job,
    urls: list[str] | None = None,
    db_path: str = job_queue.DB_PATH,
) -> None:
    """POST job result or error to the job's webhook URL (fire-and-forget).

    When explicitly enabled, uses the per-job webhook_url and falls back to
    settings.openclaw_webhook_url.
    Swallows all exceptions — a webhook failure must never affect job state.
    """
    if job.processing_mode == "local" or not settings.webhooks_enabled:
        return
    if urls is None:
        fresh = await job_queue.get_job(job.job_id, db_path=db_path)
        if fresh is not None:
            job = fresh
    destinations = urls or job.webhook_urls
    if not destinations:
        fallback = job.webhook_url or settings.openclaw_webhook_url
        destinations = [fallback] if fallback else []
    if not destinations:
        return

    if job.status == JobStatus.done:
        notion_url = (
            f"https://www.notion.so/{job.notion_page_id.replace('-', '')}"
            if job.notion_page_id
            else None
        )
        payload = {
            "status": "done",
            "job_id": job.job_id,
            "title": job.result.title if job.result else "",
            "tldr": job.summary.tldr if job.summary else "",
            "notion_url": notion_url,
            "obsidian_note_path": job.obsidian_note_path,
            "usage": job.usage.model_dump(),
        }
    else:
        payload = {
            "status": "failed",
            "job_id": job.job_id,
            "url": job.url,
            "error": job.error or "Unknown error",
        }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            for url in dict.fromkeys(destinations):
                try:
                    resp = await client.post(url, json=payload)
                    logger.info(
                        "job_id=%s event=webhook_sent status=%d",
                        job.job_id,
                        resp.status_code,
                    )
                except Exception as e:
                    logger.warning(
                        "job_id=%s event=webhook_failed error=%r",
                        job.job_id,
                        str(e),
                    )
    except Exception as e:
        logger.warning(
            "job_id=%s event=webhook_client_failed error=%r",
            job.job_id,
            str(e),
        )


async def _check_cancelled(job: Job, db_path: str) -> bool:
    """Re-read job from DB and return True if it's been cancelled."""
    fresh = await job_queue.get_job(job.job_id, db_path=db_path)
    return fresh is None or fresh.status == JobStatus.cancelled


async def _set_stage(job: Job, stage: JobStage, db_path: str) -> bool:
    """Update the job's stage and persist it."""
    job.stage = stage
    return await job_queue.update_job(job, db_path=db_path)


async def run_job(job_id: str, db_path: str = job_queue.DB_PATH) -> None:
    """Execute the full pipeline for a job, with up to MAX_RETRIES attempts.

    Updates job state in DB throughout. Fires webhook on success or final failure.
    Checks for cancellation between pipeline stages.
    """
    job = await job_queue.get_job(job_id, db_path=db_path)
    if job is None:
        logger.error("job_id=%s event=job_not_found", job_id)
        return

    log = f"job_id={job.job_id} url={job.url[:60]!r} source=unknown"
    last_error = ""
    interrupted_stage = job.stage if job.interrupted else None
    job.interrupted = False

    if job.processing_mode not in {"cloud_public", "local"}:
        job.status = JobStatus.failed
        job.stage = JobStage.failed
        job.error = "Job has an invalid persisted processing mode."
        await job_queue.update_job(job, db_path=db_path)
        return
    if job.processing_mode == "cloud_public" and not job.external_processing_approved:
        job.status = JobStatus.failed
        job.stage = JobStage.failed
        job.error = "External AI processing was not approved for this job."
        await job_queue.update_job(job, db_path=db_path)
        return

    async def persist_usage(_usage: UsageStats) -> None:
        await job_queue.update_usage(job.job_id, job.usage, db_path=db_path)

    for attempt in range(job.retry_count, MAX_RETRIES):
        if await _check_cancelled(job, db_path):
            logger.info("job_id=%s event=job_cancelled", job.job_id)
            return

        if attempt > 0:
            backoff = _BACKOFF_SECONDS[attempt - 1]
            logger.warning(
                "%s event=job_retry attempt=%d/%d backoff=%ds",
                log,
                attempt + 1,
                MAX_RETRIES,
                backoff,
            )
            await asyncio.sleep(backoff)
            if await _check_cancelled(job, db_path):
                logger.info("job_id=%s event=job_cancelled_during_backoff", job.job_id)
                return

        job.status = JobStatus.processing
        job.retry_count = attempt
        if job.result is None:
            if not await _set_stage(job, JobStage.detecting, db_path):
                return
        else:
            if not await job_queue.update_job(job, db_path=db_path):
                return

        try:
            source_type = detect_source(job.url)
            log = f"job_id={job.job_id} url={job.url[:60]!r} source={source_type}"
            logger.info("%s event=job_started attempt=%d", log, attempt + 1)

            if await _check_cancelled(job, db_path):
                return

            result: TranscriptResult
            if job.result is not None:
                result = job.result
                logger.info("%s event=transcript_reused_after_restart", log)
            else:
                # --- Transcription stage ---
                if not await _set_stage(job, JobStage.transcribing, db_path):
                    return
                if source_type == "youtube":
                    source = YouTubeSource(youtube_api_key=settings.youtube_api_key)
                    result = await source.fetch(
                        job.url,
                        job_id=job.job_id,
                        usage=job.usage,
                        persist_usage=persist_usage,
                        processing_mode=job.processing_mode,
                    )
                elif source_type == "podcast":
                    result = await PodcastSource().fetch(
                        job.url,
                        job_id=job.job_id,
                        usage=job.usage,
                        persist_usage=persist_usage,
                        processing_mode=job.processing_mode,
                    )
                elif source_type == "media":
                    result = await MediaSource().fetch(
                        job.url,
                        job_id=job.job_id,
                        usage=job.usage,
                        persist_usage=persist_usage,
                        processing_mode=job.processing_mode,
                    )
                elif source_type == "article":
                    result = await ArticleSource().fetch(
                        job.url,
                        job_id=job.job_id,
                        usage=job.usage,
                        persist_usage=persist_usage,
                        processing_mode=job.processing_mode,
                    )
                else:
                    result = await UploadSource().fetch(
                        job.url,
                        job_id=job.job_id,
                        usage=job.usage,
                        persist_usage=persist_usage,
                        processing_mode=job.processing_mode,
                    )

                job.result = result
                if not await job_queue.update_job(job, db_path=db_path):
                    cleanup_upload(job.url)
                    return

            cleanup_upload(job.url)

            if await _check_cancelled(job, db_path):
                return

            if job.summary is not None:
                summary = job.summary
                logger.info("%s event=summary_reused_after_restart", log)
            else:
                # --- Summarization stage ---
                if not await _set_stage(job, JobStage.summarizing, db_path):
                    return
                summary = await summarize(
                    result,
                    job_id=job.job_id,
                    usage=job.usage,
                    persist_usage=persist_usage,
                    processing_mode=job.processing_mode,
                )
                job.summary = summary
                if not await job_queue.update_job(job, db_path=db_path):
                    return

            if await _check_cancelled(job, db_path):
                return

            # --- Save outputs ---
            output_saved = bool(job.obsidian_note_path or job.notion_page_id)
            notion_failure: Exception | None = None
            if settings.obsidian_vault_path and not job.obsidian_note_path:
                if not await _set_stage(job, JobStage.saving_obsidian, db_path):
                    return
                job.obsidian_note_path = await save_to_obsidian(
                    result,
                    summary,
                    settings.obsidian_vault_path,
                    retain_transcript=settings.obsidian_retain_transcript,
                    summary_model=summary_model_name(job.processing_mode),
                    added_at=job.created_at,
                    usage=job.usage,
                )
                output_saved = True
                await job_queue.update_output_paths(
                    job.job_id,
                    obsidian_note_path=job.obsidian_note_path,
                    notion_page_id=job.notion_page_id,
                    notion_error=job.notion_error,
                    db_path=db_path,
                )

            if await _check_cancelled(job, db_path):
                return

            if (
                job.processing_mode != "local"
                and settings.notion_enabled
                and not job.notion_page_id
            ):
                prior_notion_page = (
                    await job_queue.find_notion_page_for_obsidian_note(
                        job.obsidian_note_path,
                        exclude_job_id=job.job_id,
                        db_path=db_path,
                    )
                    if job.obsidian_note_path
                    else None
                )
                if prior_notion_page:
                    job.notion_page_id = prior_notion_page
                    output_saved = True
                    logger.info("%s event=notion_page_reused", log)
                elif interrupted_stage == JobStage.saving_notion:
                    job.notion_error = (
                        "Notion save was interrupted and was not replayed to avoid "
                        "creating a duplicate page."
                    )
                    logger.warning(
                        "%s event=notion_save_skipped_after_interruption",
                        log,
                    )
                else:
                    if not await _set_stage(job, JobStage.saving_notion, db_path):
                        return
                    try:
                        job.notion_page_id = await save_to_notion(
                            result, summary, job_id=job.job_id
                        )
                        job.notion_error = None
                        output_saved = True
                        await job_queue.update_output_paths(
                            job.job_id,
                            obsidian_note_path=job.obsidian_note_path,
                            notion_page_id=job.notion_page_id,
                            notion_error=None,
                            db_path=db_path,
                        )
                    except Exception as notion_error:
                        notion_failure = notion_error
                        job.notion_error = str(notion_error)
                        logger.warning(
                            "%s event=notion_save_non_blocking_failure error=%r",
                            log,
                            job.notion_error,
                        )

            if await _check_cancelled(job, db_path):
                return

            if not output_saved:
                if notion_failure is not None:
                    raise notion_failure
                raise RuntimeError("No configured output destination succeeded.")

            job.status = JobStatus.done
            job.stage = JobStage.done
            if not await job_queue.update_job(job, db_path=db_path):
                return
            logger.info("%s event=job_completed title=%r", log, result.title)
            await _notify_webhook(job, db_path=db_path)
            return

        except (UnsupportedURLError, UsageLimitError) as e:
            last_error = str(e)
            logger.error("%s event=job_non_retryable error=%r", log, last_error)
            break
        except Exception as e:
            last_error = str(e)
            logger.error(
                "%s event=attempt_failed attempt=%d/%d error=%r",
                log,
                attempt + 1,
                MAX_RETRIES,
                last_error,
            )

    job.status = JobStatus.failed
    job.stage = JobStage.failed
    job.error = last_error
    if await job_queue.update_job(job, db_path=db_path):
        logger.error("%s event=job_failed_final error=%r", log, last_error)
        cleanup_upload(job.url)
        await _notify_webhook(job, db_path=db_path)

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
from exceptions import UnsupportedURLError
from models import Job, JobStage, JobStatus, TranscriptResult
from notion_writer import save_to_notion
from obsidian_writer import save_to_obsidian
from sources.podcast import PodcastSource
from sources.youtube import YouTubeSource
from summarizer import MODEL as SUMMARY_MODEL
from summarizer import summarize

logger = logging.getLogger(__name__)

_YOUTUBE_HOSTNAMES = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}
_SPOTIFY_HOSTNAMES = {"open.spotify.com"}
_APPLE_HOSTNAMES = {"podcasts.apple.com"}

MAX_RETRIES = 3
_BACKOFF_SECONDS = [5, 10, 20]  # sleep before attempt 2, 3, 4 (never used for attempt 1)


def detect_source(url: str) -> str:
    """Return source type ('youtube' | 'podcast' | 'youtube_playlist') or raise UnsupportedURLError."""
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower().lstrip("www.")

    if hostname in {"youtube.com", "youtu.be", "m.youtube.com"}:
        qs = parse_qs(parsed.query or "")
        # Playlist URL: has 'list' param but no 'v' param, or is /playlist path
        is_playlist = "playlist" in parsed.path or ("list" in qs and "v" not in qs)
        if is_playlist:
            return "youtube_playlist"

        is_watch = "watch" in parsed.path or parsed.hostname == "youtu.be"
        has_v = "v" in qs
        if not (is_watch or has_v):
            raise UnsupportedURLError(
                "Only YouTube video URLs or playlist URLs are supported."
            )
        return "youtube"

    if hostname == "open.spotify.com":
        raise UnsupportedURLError(
            "Spotify is not currently supported. Try a YouTube URL or a direct podcast RSS/MP3 URL."
        )

    if hostname == "podcasts.apple.com" or parsed.path.lower().endswith(".mp3"):
        return "podcast"

    raise UnsupportedURLError(
        "Unsupported source. Supported: YouTube (video or playlist), "
        "Apple Podcasts, direct RSS/MP3."
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


async def _notify_webhook(job: Job) -> None:
    """POST job result or error to the job's webhook URL (fire-and-forget).

    Uses the per-job webhook_url; falls back to settings.openclaw_webhook_url.
    Swallows all exceptions — a webhook failure must never affect job state.
    """
    url = job.webhook_url or settings.openclaw_webhook_url
    if not url:
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
            resp = await client.post(url, json=payload)
        logger.info("job_id=%s event=webhook_sent status=%d", job.job_id, resp.status_code)
    except Exception as e:
        logger.warning("job_id=%s event=webhook_failed error=%r", job.job_id, str(e))


async def _check_cancelled(job: Job, db_path: str) -> bool:
    """Re-read job from DB and return True if it's been cancelled."""
    fresh = await job_queue.get_job(job.job_id, db_path=db_path)
    return fresh is not None and fresh.status == JobStatus.cancelled


async def _set_stage(job: Job, stage: JobStage, db_path: str) -> None:
    """Update the job's stage and persist it."""
    job.stage = stage
    await job_queue.update_job(job, db_path=db_path)


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

    for attempt in range(MAX_RETRIES):
        if await _check_cancelled(job, db_path):
            logger.info("job_id=%s event=job_cancelled", job.job_id)
            return

        if attempt > 0:
            backoff = _BACKOFF_SECONDS[attempt - 1]
            logger.warning(
                "%s event=job_retry attempt=%d/%d backoff=%ds",
                log, attempt + 1, MAX_RETRIES, backoff,
            )
            await asyncio.sleep(backoff)

        job.status = JobStatus.processing
        job.retry_count = attempt
        await _set_stage(job, JobStage.detecting, db_path)

        try:
            source_type = detect_source(job.url)
            log = f"job_id={job.job_id} url={job.url[:60]!r} source={source_type}"
            logger.info("%s event=job_started attempt=%d", log, attempt + 1)

            if await _check_cancelled(job, db_path):
                return

            # --- Transcription stage ---
            await _set_stage(job, JobStage.transcribing, db_path)
            result: TranscriptResult
            if source_type == "youtube":
                source = YouTubeSource(youtube_api_key=settings.youtube_api_key)
                result = await source.fetch(job.url, job_id=job.job_id)
            else:
                result = await PodcastSource().fetch(job.url, job_id=job.job_id)

            job.result = result

            if await _check_cancelled(job, db_path):
                return

            # --- Summarization stage ---
            await _set_stage(job, JobStage.summarizing, db_path)
            summary = await summarize(result, job_id=job.job_id)
            job.summary = summary

            if await _check_cancelled(job, db_path):
                return

            # --- Save outputs ---
            output_saved = False
            notion_failure: Exception | None = None
            if settings.obsidian_vault_path:
                await _set_stage(job, JobStage.saving_obsidian, db_path)
                job.obsidian_note_path = await save_to_obsidian(
                    result,
                    summary,
                    settings.obsidian_vault_path,
                    retain_transcript=settings.obsidian_retain_transcript,
                    summary_model=SUMMARY_MODEL,
                    added_at=job.created_at,
                )
                output_saved = True

            if settings.notion_enabled:
                await _set_stage(job, JobStage.saving_notion, db_path)
                try:
                    job.notion_page_id = await save_to_notion(
                        result, summary, job_id=job.job_id
                    )
                    job.notion_error = None
                    output_saved = True
                except Exception as notion_error:
                    notion_failure = notion_error
                    job.notion_error = str(notion_error)
                    logger.warning(
                        "%s event=notion_save_non_blocking_failure error=%r",
                        log,
                        job.notion_error,
                    )

            if not output_saved:
                if notion_failure is not None:
                    raise notion_failure
                raise RuntimeError("No configured output destination succeeded.")

            job.status = JobStatus.done
            job.stage = JobStage.done
            logger.info("%s event=job_completed title=%r", log, result.title)
            await job_queue.update_job(job, db_path=db_path)
            await _notify_webhook(job)
            return

        except Exception as e:
            last_error = str(e)
            logger.error(
                "%s event=attempt_failed attempt=%d/%d error=%r",
                log, attempt + 1, MAX_RETRIES, last_error,
            )

    job.status = JobStatus.failed
    job.stage = JobStage.failed
    job.error = last_error
    logger.error("%s event=job_failed_final error=%r", log, last_error)
    await job_queue.update_job(job, db_path=db_path)
    await _notify_webhook(job)

"""Pipeline orchestrator.

Coordinates: detect source → source.fetch() → summarizer.summarize() → notion.save()
Failed jobs are retried up to MAX_RETRIES times with exponential backoff.
On completion (success or final failure), fires the per-job webhook if configured.
"""

import asyncio
import logging
from urllib.parse import urlparse

import httpx

import job_queue
from config import settings
from exceptions import UnsupportedURLError
from models import Job, JobStatus, TranscriptResult
from notion_writer import save_to_notion
from sources.podcast import PodcastSource
from sources.youtube import YouTubeSource
from summarizer import summarize

logger = logging.getLogger(__name__)

_YOUTUBE_HOSTNAMES = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}
_SPOTIFY_HOSTNAMES = {"open.spotify.com"}
_APPLE_HOSTNAMES = {"podcasts.apple.com"}

MAX_RETRIES = 3
_BACKOFF_SECONDS = [5, 10, 20]  # sleep before attempt 2, 3, 4 (never used for attempt 1)


def detect_source(url: str) -> str:
    """Return source type ('youtube' | 'podcast') or raise UnsupportedURLError."""
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower().lstrip("www.")

    if hostname in {"youtube.com", "youtu.be", "m.youtube.com"}:
        # Must be a video URL, not a channel/playlist
        is_watch = "watch" in parsed.path or parsed.hostname == "youtu.be"
        has_v = "v=" in (parsed.query or "")
        if not (is_watch or has_v):
            raise UnsupportedURLError(
                "Only YouTube video URLs are supported (youtube.com/watch?v=... or youtu.be/...)."
            )
        return "youtube"

    if hostname == "open.spotify.com":
        raise UnsupportedURLError(
            "Spotify is not currently supported. Try a YouTube URL or a direct podcast RSS/MP3 URL."
        )

    if hostname == "podcasts.apple.com" or parsed.path.lower().endswith(".mp3"):
        return "podcast"

    raise UnsupportedURLError(
        "Unsupported source. Supported: YouTube (youtube.com/watch, youtu.be), "
        "Apple Podcasts, direct RSS/MP3."
    )


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


async def run_job(job_id: str, db_path: str = job_queue.DB_PATH) -> None:
    """Execute the full pipeline for a job, with up to MAX_RETRIES attempts.

    Updates job state in DB throughout. Fires webhook on success or final failure.
    """
    job = await job_queue.get_job(job_id, db_path=db_path)
    if job is None:
        logger.error("job_id=%s event=job_not_found", job_id)
        return

    log = f"job_id={job.job_id} url={job.url[:60]!r} source=unknown"
    last_error = ""

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            backoff = _BACKOFF_SECONDS[attempt - 1]
            logger.warning(
                "%s event=job_retry attempt=%d/%d backoff=%ds",
                log, attempt + 1, MAX_RETRIES, backoff,
            )
            await asyncio.sleep(backoff)

        job.status = JobStatus.processing
        job.retry_count = attempt
        await job_queue.update_job(job, db_path=db_path)

        try:
            source_type = detect_source(job.url)
            log = f"job_id={job.job_id} url={job.url[:60]!r} source={source_type}"
            logger.info("%s event=job_started attempt=%d", log, attempt + 1)

            result: TranscriptResult
            if source_type == "youtube":
                source = YouTubeSource(youtube_api_key=settings.youtube_api_key)
                result = await source.fetch(job.url, job_id=job.job_id)
            else:
                result = await PodcastSource().fetch(job.url, job_id=job.job_id)

            job.result = result

            summary = await summarize(result, job_id=job.job_id)
            job.summary = summary

            page_id = await save_to_notion(result, summary, job_id=job.job_id)
            job.notion_page_id = page_id

            job.status = JobStatus.done
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
    job.error = last_error
    logger.error("%s event=job_failed_final error=%r", log, last_error)
    await job_queue.update_job(job, db_path=db_path)
    await _notify_webhook(job)

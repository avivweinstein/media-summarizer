"""Pipeline orchestrator.

Coordinates: detect source → source.fetch() → summarizer.summarize() → notion.save()
Each step is wired in progressively across phases.
"""

import logging
from urllib.parse import urlparse

import job_queue
from config import settings
from exceptions import UnsupportedURLError
from models import JobStatus, TranscriptResult
from sources.youtube import YouTubeSource

logger = logging.getLogger(__name__)

_YOUTUBE_HOSTNAMES = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}
_SPOTIFY_HOSTNAMES = {"open.spotify.com"}
_APPLE_HOSTNAMES = {"podcasts.apple.com"}


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


async def run_job(job_id: str) -> None:
    """Execute the full pipeline for a job. Updates job state in DB throughout."""
    job = await job_queue.get_job(job_id)
    if job is None:
        logger.error("job_id=%s event=job_not_found", job_id)
        return

    log = f"job_id={job.job_id} url={job.url[:60]!r} source=unknown"

    # Mark as processing
    job.status = JobStatus.processing
    await job_queue.update_job(job)

    try:
        source_type = detect_source(job.url)
        log = f"job_id={job.job_id} url={job.url[:60]!r} source={source_type}"
        logger.info("%s event=job_started", log)

        result: TranscriptResult

        if source_type == "youtube":
            source = YouTubeSource(youtube_api_key=settings.youtube_api_key)
            result = await source.fetch(job.url, job_id=job.job_id)
        else:
            # podcast — Phase 5
            raise UnsupportedURLError("Podcast support is not yet implemented.")

        # Phase 2: store transcript result; summarization (Phase 3) and Notion (Phase 4) TBD
        job.status = JobStatus.done
        job.result = result
        logger.info("%s event=job_completed title=%r", log, result.title)

    except Exception as e:
        job.status = JobStatus.failed
        job.error = str(e)
        logger.error("%s event=job_failed error=%r", log, str(e))

    await job_queue.update_job(job)

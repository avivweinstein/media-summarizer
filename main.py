"""FastAPI application — entry point.

Routes:
  POST  /summarize           body: { url, webhook_url? }  -> { job_id }
  POST  /summarize/bulk      body: { urls, webhook_url? } -> { job_ids }
  GET   /job/{job_id}        -> { status, result?, error? }
  GET   /jobs                -> list of recent jobs (for web UI)
  POST  /job/{job_id}/cancel -> 200
  DELETE /job/{job_id}       -> 200
  DELETE /jobs/failed         -> { deleted: N }
  GET   /jobs/stream         -> SSE stream of job updates
  GET   /health              -> 200 (shallow) or ?deep=true for full check
  GET   /                    -> web UI dashboard
"""

import asyncio
import logging
import logging.config
import tomllib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

import job_queue
from config import settings
from exceptions import UnsupportedURLError
from models import (
    BulkSummarizeRequest,
    BulkSummarizeResponse,
    JobResponse,
    JobStage,
    JobStatus,
    SummarizeRequest,
    SummarizeResponse,
)
from pipeline import detect_source, expand_playlist
from transcriber import ensure_tmp_dir
from worker import job_worker

logger = logging.getLogger(__name__)

# How often the SSE endpoint polls the DB for changes (seconds)
_SSE_POLL_INTERVAL = 2.0

# Auto-cleanup: delete completed/failed/cancelled jobs older than this many days
_JOB_TTL_DAYS = 90


def _configure_logging() -> None:
    toml_path = Path(__file__).parent / "logging.toml"
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)
    logging.config.dictConfig(config)


async def _cleanup_old_jobs() -> None:
    """Delete jobs older than _JOB_TTL_DAYS on startup."""
    deleted = await job_queue.delete_old_jobs(max_age_days=_JOB_TTL_DAYS)
    if deleted:
        logger.info("job_id=- url=- source=- event=ttl_cleanup deleted=%d", deleted)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _configure_logging()
    await job_queue.init_db()
    ensure_tmp_dir()
    await _cleanup_old_jobs()
    job_worker.start()
    logger.info("job_id=- url=- source=- event=server_started")
    yield
    await job_worker.stop()
    logger.info("job_id=- url=- source=- event=server_stopped")


app = FastAPI(title="Media Summarizer", lifespan=lifespan)


def _job_to_response(job: job_queue.Job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        url=job.url,
        status=job.status,
        stage=job.stage,
        created_at=job.created_at,
        updated_at=job.updated_at,
        retry_count=job.retry_count,
        result=job.result,
        summary=job.summary,
        notion_page_id=job.notion_page_id,
        error=job.error,
        parent_job_id=job.parent_job_id,
    )


# ---------------------------------------------------------------------------
# Submit endpoints
# ---------------------------------------------------------------------------

@app.post("/summarize", response_model=SummarizeResponse, status_code=202)
async def submit_url(request: SummarizeRequest) -> SummarizeResponse:
    """Validate URL, enqueue job, return job_id immediately.

    Accepts single video/podcast URLs and playlist URLs. Playlists are expanded
    into individual jobs; the first job_id is returned.
    """
    url = request.url.strip()
    try:
        source_type = detect_source(url)
    except UnsupportedURLError as e:
        logger.warning("job_id=- url=%r source=- event=url_rejected reason=%r", url[:60], str(e))
        raise HTTPException(status_code=400, detail=str(e))

    if source_type == "youtube_playlist":
        try:
            video_urls = await expand_playlist(url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to expand playlist: {e}")
        if not video_urls:
            raise HTTPException(status_code=400, detail="Playlist is empty or could not be read.")

        # Create a job for each video
        first_job_id = ""
        for video_url in video_urls:
            job = await job_queue.create_job(video_url, request.webhook_url)
            if not first_job_id:
                first_job_id = job.job_id
            await job_worker.enqueue(job.job_id)
        return SummarizeResponse(job_id=first_job_id)

    job = await job_queue.create_job(url, request.webhook_url)
    await job_worker.enqueue(job.job_id)
    return SummarizeResponse(job_id=job.job_id)


@app.post("/summarize/bulk", response_model=BulkSummarizeResponse, status_code=202)
async def submit_bulk(request: BulkSummarizeRequest) -> BulkSummarizeResponse:
    """Submit multiple URLs at once. Each gets its own job."""
    job_ids: list[str] = []
    errors: list[str] = []

    for url in request.urls:
        url = url.strip()
        try:
            source_type = detect_source(url)
        except UnsupportedURLError as e:
            errors.append(f"{url[:60]}: {e}")
            continue

        if source_type == "youtube_playlist":
            try:
                video_urls = await expand_playlist(url)
            except Exception as e:
                errors.append(f"{url[:60]}: playlist expansion failed: {e}")
                continue
            for video_url in video_urls:
                job = await job_queue.create_job(video_url, request.webhook_url)
                job_ids.append(job.job_id)
                await job_worker.enqueue(job.job_id)
        else:
            job = await job_queue.create_job(url, request.webhook_url)
            job_ids.append(job.job_id)
            await job_worker.enqueue(job.job_id)

    if not job_ids and errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    return BulkSummarizeResponse(job_ids=job_ids)


# ---------------------------------------------------------------------------
# Job management endpoints
# ---------------------------------------------------------------------------

@app.get("/job/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Return the current state of a job."""
    job = await job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return _job_to_response(job)


@app.get("/jobs", response_model=list[JobResponse])
async def list_jobs() -> list[JobResponse]:
    """Return the 50 most recent jobs."""
    jobs = await job_queue.list_jobs(limit=50)
    return [_job_to_response(j) for j in jobs]


@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, str]:
    """Cancel a pending or processing job."""
    job = await job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job.status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
        raise HTTPException(
            status_code=409,
            detail=f"Job is already {job.status.value} and cannot be cancelled.",
        )
    job.status = JobStatus.cancelled
    job.stage = JobStage.failed
    await job_queue.update_job(job)
    logger.info("job_id=%s url=%.60s source=- event=job_cancelled_by_user", job.job_id, job.url)
    return {"status": "cancelled"}


@app.delete("/job/{job_id}")
async def delete_job(job_id: str) -> dict[str, str]:
    """Delete a single job by ID."""
    deleted = await job_queue.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return {"status": "deleted"}


@app.delete("/jobs/failed")
async def delete_failed_jobs() -> dict[str, int]:
    """Delete all failed jobs."""
    count = await job_queue.delete_jobs_by_status("failed")
    return {"deleted": count}


@app.delete("/jobs/cancelled")
async def delete_cancelled_jobs() -> dict[str, int]:
    """Delete all cancelled jobs."""
    count = await job_queue.delete_jobs_by_status("cancelled")
    return {"deleted": count}


# ---------------------------------------------------------------------------
# SSE: real-time job updates
# ---------------------------------------------------------------------------

@app.get("/jobs/stream")
async def jobs_stream() -> EventSourceResponse:
    """Server-Sent Events stream of job list updates.

    Polls the DB every _SSE_POLL_INTERVAL seconds and emits a JSON array of
    all jobs whenever any job's updated_at has changed.
    """

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        last_snapshot: str = ""
        while True:
            try:
                jobs = await job_queue.list_jobs(limit=50)
                responses = [_job_to_response(j) for j in jobs]
                import json
                snapshot = json.dumps(
                    [r.model_dump(mode="json") for r in responses],
                    default=str,
                )
                if snapshot != last_snapshot:
                    last_snapshot = snapshot
                    yield {"event": "jobs", "data": snapshot}
            except Exception:
                pass
            await asyncio.sleep(_SSE_POLL_INTERVAL)

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(deep: bool = Query(False)) -> dict[str, object]:
    """Health check. Pass ?deep=true to verify API keys and DB connectivity."""
    if not deep:
        return {"status": "ok"}

    checks: dict[str, object] = {"status": "ok"}
    errors: list[str] = []

    # DB check
    try:
        await job_queue.list_jobs(limit=1)
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {e}"
        errors.append("db")

    # Anthropic API key check
    if settings.anthropic_api_key:
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=settings.anthropic_api_key)
            await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            checks["anthropic"] = "ok"
        except Exception as e:
            checks["anthropic"] = f"error: {e}"
            errors.append("anthropic")
    else:
        checks["anthropic"] = "not configured"
        errors.append("anthropic")

    # OpenAI API key check (just validates the key, no actual transcription)
    if settings.openai_api_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            await client.models.list()
            checks["openai"] = "ok"
        except Exception as e:
            checks["openai"] = f"error: {e}"
            errors.append("openai")
    else:
        checks["openai"] = "not configured"
        errors.append("openai")

    # Notion API key check
    if settings.notion_api_key:
        try:
            from notion_client import AsyncClient
            client = AsyncClient(auth=settings.notion_api_key)
            await client.databases.retrieve(database_id=settings.notion_database_id)
            checks["notion"] = "ok"
        except Exception as e:
            checks["notion"] = f"error: {e}"
            errors.append("notion")
    else:
        checks["notion"] = "not configured"
        errors.append("notion")

    # Worker status
    checks["worker_queue_size"] = job_worker.queue_size

    if errors:
        checks["status"] = "degraded"

    return checks


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the job status dashboard."""
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return HTMLResponse(content=html)

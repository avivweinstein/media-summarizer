"""FastAPI application — entry point.

Routes:
  POST  /summarize           body: { url, webhook_url? }  → { job_id }
  GET   /job/{job_id}        → { status, result?, error? }
  GET   /jobs                → list of recent jobs (for web UI)
  GET   /health              → 200
  GET   /                    → simple web UI (Phase 6)
"""

import logging
import logging.config
import tomllib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

import job_queue
from exceptions import UnsupportedURLError
from models import JobResponse, SummarizeRequest, SummarizeResponse
from pipeline import detect_source, run_job
from transcriber import ensure_tmp_dir

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    toml_path = Path(__file__).parent / "logging.toml"
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)
    logging.config.dictConfig(config)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _configure_logging()
    await job_queue.init_db()
    ensure_tmp_dir()
    logger.info("job_id=- url=- source=- event=server_started")
    yield
    logger.info("job_id=- url=- source=- event=server_stopped")


app = FastAPI(title="Media Summarizer", lifespan=lifespan)


def _job_to_response(job: Any) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        url=job.url,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        retry_count=job.retry_count,
        result=job.result,
        summary=job.summary,
        notion_page_id=job.notion_page_id,
        error=job.error,
    )


@app.post("/summarize", response_model=SummarizeResponse, status_code=202)
async def submit_url(
    request: SummarizeRequest, background_tasks: BackgroundTasks
) -> SummarizeResponse:
    """Validate URL, enqueue job, return job_id immediately."""
    try:
        detect_source(request.url)
    except UnsupportedURLError as e:
        logger.warning("job_id=- url=%r source=- event=url_rejected reason=%r", request.url[:60], str(e))
        raise HTTPException(status_code=400, detail=str(e))

    job = await job_queue.create_job(request.url, request.webhook_url)
    background_tasks.add_task(run_job, job.job_id)
    return SummarizeResponse(job_id=job.job_id)


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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Job dashboard — Phase 6."""
    raise HTTPException(status_code=501, detail="Web UI coming in Phase 6.")

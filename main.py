"""FastAPI application — entry point.

Routes:
  POST  /summarize           body: { url, webhook_url? }  → { job_id }
  GET   /job/{job_id}        → { status, result?, error? }
  GET   /jobs                → list of recent jobs (for web UI)
  GET   /health              → 200
  GET   /                    → simple web UI (job dashboard)
"""

import logging
import logging.config
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

import job_queue
from models import JobResponse, JobStatus, SummarizeRequest, SummarizeResponse

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    toml_path = Path(__file__).parent / "logging.toml"
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)
    logging.config.dictConfig(config)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    _configure_logging()
    await job_queue.init_db()
    logger.info("job_id=- url=- source=- event=server_started")
    yield
    logger.info("job_id=- url=- source=- event=server_stopped")


app = FastAPI(title="Media Summarizer", lifespan=lifespan)


@app.post("/summarize", response_model=SummarizeResponse, status_code=202)
async def submit_url(request: SummarizeRequest) -> SummarizeResponse:
    """Accept a URL and enqueue a summarization job."""
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.get("/job/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Return the current state of a job."""
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.get("/jobs", response_model=list[JobResponse])
async def list_jobs() -> list[JobResponse]:
    """Return the 50 most recent jobs (used by the web UI)."""
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Simple job dashboard — served as plain HTML."""
    raise HTTPException(status_code=501, detail="Not implemented yet")

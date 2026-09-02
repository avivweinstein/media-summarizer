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
import fcntl
import json
import logging
import logging.config
import os
import shutil
import tomllib
import uuid
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

import job_queue
from config import settings
from exceptions import UnsupportedURLError
from library import ask_library, search_library
from models import (
    BulkSummarizeRequest,
    BulkSummarizeResponse,
    Job,
    JobResponse,
    JobStatus,
    LibraryAnswer,
    LibraryAskRequest,
    LibrarySearchHit,
    SummarizeRequest,
    SummarizeResponse,
)
from pipeline import _notify_webhook, detect_source, expand_playlist
from sources.upload import UPLOAD_EXTENSIONS, cleanup_upload, reconcile_uploads, upload_path
from transcriber import ensure_tmp_dir
from worker import job_worker

logger = logging.getLogger(__name__)

# How often the SSE endpoint polls the DB for changes (seconds)
_SSE_POLL_INTERVAL = 2.0

# Auto-cleanup: delete completed/failed/cancelled jobs older than this many days
_JOB_TTL_DAYS = 90
_INSTANCE_LOCK_PATH = Path("/tmp/media-summarizer/service.lock")


@contextmanager
def _single_instance_lock(path: Path = _INSTANCE_LOCK_PATH) -> Iterator[None]:
    """Prevent multiple processes from recovering and executing the same jobs."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.fchmod(fd, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another media-summarizer process already owns the job queue."
            ) from error
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _obsidian_destinations_writable(vault_path: Path, retain_transcript: bool) -> bool:
    destinations = [vault_path / "Generated" / "Summaries"]
    if retain_transcript:
        destinations.append(vault_path / "Generated" / "Transcripts")

    for destination in destinations:
        existing = destination
        while not existing.exists() and existing != vault_path:
            existing = existing.parent
        if not existing.is_dir() or not os.access(existing, os.W_OK | os.X_OK):
            return False
    return True


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
    with _single_instance_lock():
        _configure_logging()
        await job_queue.init_db()
        ensure_tmp_dir()
        await _cleanup_old_jobs()
        removed_uploads = reconcile_uploads(await job_queue.list_active_upload_urls())
        if removed_uploads:
            logger.warning(
                "job_id=- url=- source=upload event=startup_cleanup removed_files=%d",
                removed_uploads,
            )
        recovered_job_ids = await job_queue.recover_incomplete_jobs()
        job_worker.start()
        for job_id in recovered_job_ids:
            await job_worker.enqueue(job_id)
        if recovered_job_ids:
            logger.info(
                "job_id=- url=- source=- event=jobs_recovered count=%d",
                len(recovered_job_ids),
            )
        logger.info("job_id=- url=- source=- event=server_started")
        yield
        await job_worker.stop()
        logger.info("job_id=- url=- source=- event=server_stopped")


app = FastAPI(title="Media Summarizer", lifespan=lifespan)


async def _create_and_enqueue(
    url: str,
    webhook_url: str | None,
    *,
    processing_mode: str | None = None,
    external_processing_approved: bool = False,
) -> Job:
    mode = processing_mode or settings.processing_mode
    job, created = await job_queue.create_or_get_job(
        url,
        webhook_url,
        processing_mode=mode,
        external_processing_approved=external_processing_approved,
    )
    if created:
        await job_worker.enqueue(job.job_id)
    else:
        logger.info(
            "job_id=%s url=%.60s source=- event=duplicate_submission_reused",
            job.job_id,
            url,
        )
        if job.status == JobStatus.done and webhook_url:
            await _notify_webhook(job, urls=[webhook_url])
    return job


async def _persist_upload(file: UploadFile) -> str:
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.casefold()
    if suffix not in UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Supported uploads: txt, md, mp3, m4a, wav, mp4, mov, and webm.",
        )
    upload_id = str(uuid.uuid4())
    url = f"upload://{upload_id}/{quote(filename)}"
    destination = upload_path(url)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.chmod(0o700)
    max_bytes = (
        settings.max_article_download_bytes
        if suffix in {".md", ".txt"}
        else settings.max_audio_download_bytes
    )
    written = 0
    try:
        with open(destination, "xb") as output:
            os.fchmod(output.fileno(), 0o600)
            while chunk := await file.read(64 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Upload exceeds the configured size limit.",
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return url


def _job_to_response(job: Job, *, include_transcript: bool = True) -> JobResponse:
    result = job.result
    if result is not None and not include_transcript:
        result = result.model_copy(update={"transcript": "", "segments": []})
    return JobResponse(
        job_id=job.job_id,
        url=job.url,
        status=job.status,
        stage=job.stage,
        created_at=job.created_at,
        updated_at=job.updated_at,
        retry_count=job.retry_count,
        interruption_count=job.interruption_count,
        result=result,
        summary=job.summary,
        notion_page_id=job.notion_page_id,
        notion_error=job.notion_error,
        obsidian_note_path=job.obsidian_note_path,
        error=job.error,
        parent_job_id=job.parent_job_id,
        usage=job.usage,
        processing_mode=job.processing_mode,
    )


def _require_processing_approval(approved: bool) -> None:
    if settings.processing_mode == "cloud_public" and not approved:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cloud mode requires explicit confirmation that the content is public or "
                "approved for external AI processing."
            ),
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
    _require_processing_approval(request.external_processing_approved)
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
            job = await _create_and_enqueue(
                video_url,
                request.webhook_url,
                processing_mode=settings.processing_mode,
                external_processing_approved=request.external_processing_approved,
            )
            if not first_job_id:
                first_job_id = job.job_id
        return SummarizeResponse(job_id=first_job_id)

    job = await _create_and_enqueue(
        url,
        request.webhook_url,
        processing_mode=settings.processing_mode,
        external_processing_approved=request.external_processing_approved,
    )
    return SummarizeResponse(job_id=job.job_id)


@app.post("/summarize/upload", response_model=SummarizeResponse, status_code=202)
async def submit_upload(
    file: UploadFile = File(...),
    external_processing_approved: bool = Form(False),
    webhook_url: str | None = Form(None),
) -> SummarizeResponse:
    """Persist and enqueue an upload after an explicit data-boundary acknowledgement."""
    _require_processing_approval(external_processing_approved)
    url = await _persist_upload(file)
    try:
        job = await _create_and_enqueue(
            url,
            webhook_url,
            processing_mode=settings.processing_mode,
            external_processing_approved=external_processing_approved,
        )
    except Exception:
        cleanup_upload(url)
        raise
    return SummarizeResponse(job_id=job.job_id)


@app.post("/summarize/bulk", response_model=BulkSummarizeResponse, status_code=202)
async def submit_bulk(request: BulkSummarizeRequest) -> BulkSummarizeResponse:
    """Submit multiple URLs at once. Each gets its own job."""
    _require_processing_approval(request.external_processing_approved)
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
                job = await _create_and_enqueue(
                    video_url,
                    request.webhook_url,
                    processing_mode=settings.processing_mode,
                    external_processing_approved=request.external_processing_approved,
                )
                if job.job_id not in job_ids:
                    job_ids.append(job.job_id)
        else:
            job = await _create_and_enqueue(
                url,
                request.webhook_url,
                processing_mode=settings.processing_mode,
                external_processing_approved=request.external_processing_approved,
            )
            if job.job_id not in job_ids:
                job_ids.append(job.job_id)

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
    return [_job_to_response(j, include_transcript=False) for j in jobs]


@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, str]:
    """Cancel a pending or processing job."""
    job = await job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if not await job_queue.mark_job_cancelled(job_id):
        current = await job_queue.get_job(job_id)
        status = current.status.value if current else "deleted"
        raise HTTPException(
            status_code=409,
            detail=f"Job is already {status} and cannot be cancelled.",
        )
    if not settings.db_retain_transcript:
        await job_queue.redact_job_transcript(job_id)
    await job_worker.cancel(job_id)
    cleanup_upload(job.url)
    logger.info("job_id=%s url=%.60s source=- event=job_cancelled_by_user", job.job_id, job.url)
    return {"status": "cancelled"}


@app.delete("/job/{job_id}")
async def delete_job(job_id: str) -> dict[str, str]:
    """Delete a single job by ID."""
    job = await job_queue.get_job(job_id)
    deleted = await job_queue.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job is not None:
        cleanup_upload(job.url)
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
                responses = [_job_to_response(j, include_transcript=False) for j in jobs]
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


@app.get("/library/search", response_model=list[LibrarySearchHit])
async def library_search(
    q: str = Query(min_length=2, max_length=500),
    limit: int = Query(default=10, ge=1, le=20),
) -> list[LibrarySearchHit]:
    """Search generated Obsidian summaries and transcripts locally."""
    if not settings.obsidian_vault_path:
        raise HTTPException(status_code=503, detail="Obsidian vault is not configured.")
    try:
        return await search_library(settings.obsidian_vault_path, q, limit)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/library/ask", response_model=LibraryAnswer)
async def library_ask(request: LibraryAskRequest) -> LibraryAnswer:
    """Answer from local library excerpts; never sends library text to cloud AI."""
    if not settings.obsidian_vault_path:
        raise HTTPException(status_code=503, detail="Obsidian vault is not configured.")
    try:
        return await ask_library(
            settings.obsidian_vault_path,
            request.question,
            limit=request.limit,
            provider=settings.library_qa_provider,
            ollama_model=settings.ollama_model,
            ollama_base_url=settings.ollama_base_url,
        )
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


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

    # AI provider checks
    if settings.processing_mode == "local":
        checks["anthropic"] = "disabled (local mode)"
        checks["openai"] = "disabled (local mode)"
        try:
            parsed_ollama = urlparse(settings.ollama_base_url)
            if parsed_ollama.scheme != "http" or parsed_ollama.hostname not in {
                "127.0.0.1",
                "::1",
                "localhost",
            }:
                raise ValueError("Ollama must use a loopback HTTP endpoint.")
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
                response.raise_for_status()
                payload = response.json()
                models = payload.get("models", []) if isinstance(payload, dict) else []
                installed = {
                    str(item.get("name") or item.get("model"))
                    for item in models
                    if isinstance(item, dict)
                }
                if settings.ollama_model not in installed:
                    raise ValueError(
                        f"Configured Ollama model {settings.ollama_model!r} is not installed."
                    )
            checks["ollama"] = "ok"
        except Exception as e:
            checks["ollama"] = f"error: {e}"
            errors.append("ollama")
        local_whisper = shutil.which(settings.local_whisper_executable)
        local_model = Path(settings.local_whisper_model).expanduser()
        if local_whisper and local_model.is_file():
            checks["local_whisper"] = "ok"
        else:
            checks["local_whisper"] = "not configured"
            errors.append("local_whisper")
    elif settings.anthropic_api_key:
        try:
            from anthropic import AsyncAnthropic

            anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
            await anthropic_client.messages.create(
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
    if settings.processing_mode == "local":
        pass
    elif settings.openai_api_key:
        try:
            from openai import AsyncOpenAI

            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            await openai_client.models.list()
            checks["openai"] = "ok"
        except Exception as e:
            checks["openai"] = f"error: {e}"
            errors.append("openai")
    else:
        checks["openai"] = "not configured"
        errors.append("openai")

    # Obsidian vault check
    if settings.obsidian_vault_path:
        vault_path = Path(settings.obsidian_vault_path).expanduser()
        if vault_path.is_dir() and (vault_path / ".obsidian").is_dir():
            writable = _obsidian_destinations_writable(
                vault_path,
                settings.obsidian_retain_transcript,
            )
            checks["obsidian"] = "ok" if writable else "not writable"
            if checks["obsidian"] != "ok":
                errors.append("obsidian")
        else:
            checks["obsidian"] = "invalid vault"
            errors.append("obsidian")
    else:
        checks["obsidian"] = "not configured"

    # Notion API key check
    if settings.processing_mode == "local":
        checks["notion"] = "disabled (local mode)"
    elif not settings.notion_enabled:
        checks["notion"] = "disabled"
    elif settings.notion_api_key:
        try:
            from notion_client import AsyncClient

            notion_client = AsyncClient(auth=settings.notion_api_key)
            await notion_client.databases.retrieve(database_id=settings.notion_database_id)
            checks["notion"] = "ok"
        except Exception as e:
            checks["notion"] = f"error: {e}"
            errors.append("notion")
    else:
        checks["notion"] = "not configured"
        errors.append("notion")

    if not settings.obsidian_vault_path and (
        settings.processing_mode == "local" or not settings.notion_enabled
    ):
        checks["storage"] = "no destination configured"
        errors.append("storage")

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
    if settings.processing_mode == "local":
        mode_label = "Local-only mode"
        boundary_message = (
            "Local-only mode: transcripts stay on this Mac; Anthropic, OpenAI, "
            "Notion, and webhooks are disabled."
        )
    else:
        mode_label = "Cloud-public mode"
        boundary_message = (
            "Cloud-public mode: submit only public content or material explicitly "
            "approved for external AI processing."
        )
    html = html.replace("__PROCESSING_MODE_LABEL__", mode_label)
    html = html.replace("__DATA_BOUNDARY_MESSAGE__", boundary_message)
    return HTMLResponse(content=html)

"""SQLite-backed async job queue.

Job states: pending → processing → done | failed
Failed jobs are retried up to MAX_RETRIES times with exponential backoff.
Jobs survive server restarts because they're persisted in SQLite.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

from models import Job, JobStage, JobStatus, Summary, TranscriptResult

logger = logging.getLogger(__name__)

DB_PATH = "jobs.db"
MAX_RETRIES = 3


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _serialize_job(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "url": job.url,
        "status": job.status.value,
        "stage": job.stage.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "retry_count": job.retry_count,
        "result": job.result.model_dump_json() if job.result else None,
        "summary": job.summary.model_dump_json() if job.summary else None,
        "notion_page_id": job.notion_page_id,
        "notion_error": job.notion_error,
        "obsidian_note_path": job.obsidian_note_path,
        "error": job.error,
        "webhook_url": job.webhook_url,
        "parent_job_id": job.parent_job_id,
    }


def _deserialize_job(row: aiosqlite.Row) -> Job:
    data = dict(row)
    result = None
    if data["result"]:
        result = TranscriptResult.model_validate_json(data["result"])
    summary = None
    if data["summary"]:
        summary = Summary.model_validate_json(data["summary"])
    # Handle DB rows created before the stage/parent_job_id columns existed
    raw_stage = data.get("stage") or "queued"
    return Job(
        job_id=data["job_id"],
        url=data["url"],
        status=JobStatus(data["status"]),
        stage=JobStage(raw_stage),
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        retry_count=data["retry_count"],
        result=result,
        summary=summary,
        notion_page_id=data["notion_page_id"],
        notion_error=data.get("notion_error"),
        obsidian_note_path=data.get("obsidian_note_path"),
        error=data["error"],
        webhook_url=data["webhook_url"],
        parent_job_id=data.get("parent_job_id"),
    )


async def init_db(db_path: str = DB_PATH) -> None:
    """Create the jobs table if it doesn't exist, and run schema migrations."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id      TEXT PRIMARY KEY,
                url         TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                stage       TEXT NOT NULL DEFAULT 'queued',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                result      TEXT,
                summary     TEXT,
                notion_page_id TEXT,
                notion_error TEXT,
                obsidian_note_path TEXT,
                error       TEXT,
                webhook_url TEXT,
                parent_job_id TEXT
            )
        """)
        # Migrate: add columns if they don't exist (for existing DBs)
        cursor = await db.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "stage" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN stage TEXT NOT NULL DEFAULT 'queued'")
        if "parent_job_id" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN parent_job_id TEXT")
        if "notion_error" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN notion_error TEXT")
        if "obsidian_note_path" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN obsidian_note_path TEXT")
        await db.commit()


async def create_job(
    url: str,
    webhook_url: str | None = None,
    parent_job_id: str | None = None,
    db_path: str = DB_PATH,
) -> Job:
    """Insert a new pending job and return it."""
    now = _utcnow()
    job = Job(
        job_id=str(uuid.uuid4()),
        url=url,
        status=JobStatus.pending,
        stage=JobStage.queued,
        created_at=now,
        updated_at=now,
        webhook_url=webhook_url,
        parent_job_id=parent_job_id,
    )
    data = _serialize_job(job)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO jobs
                (job_id, url, status, stage, created_at, updated_at, retry_count,
                 result, summary, notion_page_id, notion_error, obsidian_note_path,
                 error, webhook_url, parent_job_id)
            VALUES
                (:job_id, :url, :status, :stage, :created_at, :updated_at, :retry_count,
                 :result, :summary, :notion_page_id, :notion_error, :obsidian_note_path,
                 :error, :webhook_url, :parent_job_id)
            """,
            data,
        )
        await db.commit()
    logger.info("job_id=%s url=%.60s source=unknown event=job_created", job.job_id, url)
    return job


async def get_job(job_id: str, db_path: str = DB_PATH) -> Job | None:
    """Fetch a single job by ID."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    return _deserialize_job(row)


async def list_jobs(limit: int = 50, db_path: str = DB_PATH) -> list[Job]:
    """Return the most recent `limit` jobs, newest first."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
    return [_deserialize_job(row) for row in rows]


async def update_job(job: Job, db_path: str = DB_PATH) -> None:
    """Persist all mutable fields of a job back to SQLite."""
    job.updated_at = _utcnow()
    data = _serialize_job(job)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE jobs SET
                status        = :status,
                stage         = :stage,
                updated_at    = :updated_at,
                retry_count   = :retry_count,
                result        = :result,
                summary       = :summary,
                notion_page_id = :notion_page_id,
                notion_error   = :notion_error,
                obsidian_note_path = :obsidian_note_path,
                error         = :error
            WHERE job_id = :job_id
            """,
            data,
        )
        await db.commit()


async def delete_job(job_id: str, db_path: str = DB_PATH) -> bool:
    """Delete a single job by ID. Returns True if a row was deleted."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        await db.commit()
        return cursor.rowcount > 0


async def delete_jobs_by_status(status: str, db_path: str = DB_PATH) -> int:
    """Delete all jobs with the given status. Returns count of deleted rows."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("DELETE FROM jobs WHERE status = ?", (status,))
        await db.commit()
        return cursor.rowcount


async def delete_old_jobs(max_age_days: int = 90, db_path: str = DB_PATH) -> int:
    """Delete jobs older than max_age_days. Returns count of deleted rows."""
    cutoff = (_utcnow() - timedelta(days=max_age_days)).isoformat()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "DELETE FROM jobs WHERE created_at < ? AND status IN ('done', 'failed', 'cancelled')",
            (cutoff,),
        )
        await db.commit()
        return cursor.rowcount


async def claim_next_pending_job(db_path: str = DB_PATH) -> Job | None:
    """Atomically move one pending job to processing. Returns it or None."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        # Fetch the oldest pending job
        async with db.execute(
            "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        job = _deserialize_job(row)
        now = _utcnow().isoformat()
        await db.execute(
            "UPDATE jobs SET status = 'processing', updated_at = ? WHERE job_id = ? AND status = 'pending'",
            (now, job.job_id),
        )
        await db.commit()
    # Re-fetch to confirm the claim succeeded (another worker could have claimed it)
    claimed = await get_job(job.job_id, db_path)
    if claimed and claimed.status == JobStatus.processing:
        return claimed
    return None

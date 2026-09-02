"""SQLite-backed async job queue.

Job states: pending → processing → done | failed
Failed jobs are retried up to MAX_RETRIES times with exponential backoff.
Jobs survive server restarts because they're persisted in SQLite.
"""

import json
import logging
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from config import settings
from models import Job, JobStage, JobStatus, Summary, TranscriptResult, UsageStats
from url_identity import submission_identity

logger = logging.getLogger(__name__)

DB_PATH = str(Path(settings.job_db_path).expanduser().resolve())
LEGACY_DB_PATH = Path(__file__).with_name("jobs.db")
MAX_RETRIES = 3
MAX_INTERRUPTION_RECOVERIES = 3


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
        "webhook_urls": json.dumps(job.webhook_urls),
        "parent_job_id": job.parent_job_id,
        "dedupe_key": job.dedupe_key,
        "usage": job.usage.model_dump_json(),
        "interrupted": int(job.interrupted),
        "interruption_count": job.interruption_count,
        "processing_mode": job.processing_mode,
        "external_processing_approved": int(job.external_processing_approved),
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
    webhook_urls = (
        json.loads(data["webhook_urls"])
        if data.get("webhook_urls")
        else ([data["webhook_url"]] if data.get("webhook_url") else [])
    )
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
        webhook_urls=webhook_urls,
        parent_job_id=data.get("parent_job_id"),
        dedupe_key=data.get("dedupe_key"),
        usage=(
            UsageStats.model_validate_json(data["usage"]) if data.get("usage") else UsageStats()
        ),
        interrupted=bool(data.get("interrupted", 0)),
        interruption_count=int(data.get("interruption_count", 0)),
        processing_mode=data.get("processing_mode") or "cloud_public",
        external_processing_approved=bool(data.get("external_processing_approved", 0)),
    )


def _without_transcript(result: TranscriptResult) -> TranscriptResult:
    return result.model_copy(update={"transcript": "", "segments": []})


def _archive_id(job: Job) -> str:
    output_id = job.obsidian_note_path or f"notion:{job.notion_page_id}"
    return f"{job.processing_mode}:{output_id}"


def _obsidian_artifact_exists(note_path: str | None) -> bool:
    if note_path is None:
        return True
    if not settings.obsidian_vault_path:
        return False
    vault = Path(settings.obsidian_vault_path).expanduser().resolve()
    candidate = (vault / note_path).resolve()
    return candidate.is_relative_to(vault) and candidate.is_file()


async def _upsert_archive_row(db: aiosqlite.Connection, job: Job) -> None:
    if (
        not job.dedupe_key
        or job.result is None
        or job.summary is None
        or not (job.obsidian_note_path or job.notion_page_id)
    ):
        raise ValueError("A completed output is required to record archive identity.")
    now = _utcnow().isoformat()
    await db.execute(
        """INSERT INTO archive_entries
               (archive_id, dedupe_key, processing_mode, obsidian_note_path,
                notion_page_id, result, summary, archived_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(archive_id) DO UPDATE SET
               dedupe_key = excluded.dedupe_key,
               notion_page_id = COALESCE(excluded.notion_page_id, archive_entries.notion_page_id),
               result = excluded.result,
               summary = excluded.summary,
               updated_at = excluded.updated_at""",
        (
            _archive_id(job),
            job.dedupe_key,
            job.processing_mode,
            job.obsidian_note_path,
            job.notion_page_id,
            _without_transcript(job.result).model_dump_json(),
            job.summary.model_dump_json(),
            job.created_at.isoformat(),
            now,
        ),
    )


async def init_db(db_path: str = DB_PATH) -> None:
    """Create the jobs table if it doesn't exist, and run schema migrations."""
    path = Path(db_path).expanduser().resolve()
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent_existed:
        path.parent.chmod(0o700)
    if path == Path(DB_PATH) and not path.exists() and LEGACY_DB_PATH.is_file():
        shutil.copy2(LEGACY_DB_PATH, path)
        logger.info("Migrated legacy job database to %s", path)
    db_path = str(path)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if not settings.db_retain_transcript:
            await db.execute("PRAGMA secure_delete = ON")
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
                webhook_urls TEXT,
                parent_job_id TEXT,
                dedupe_key TEXT,
                usage TEXT,
                interrupted INTEGER NOT NULL DEFAULT 0,
                interruption_count INTEGER NOT NULL DEFAULT 0,
                processing_mode TEXT NOT NULL DEFAULT 'nvidia_internal',
                external_processing_approved INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Migrate: add columns if they don't exist (for existing DBs)
        cursor = await db.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "stage" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN stage TEXT NOT NULL DEFAULT 'queued'")
        if "parent_job_id" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN parent_job_id TEXT")
        if "webhook_urls" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN webhook_urls TEXT")
        if "notion_error" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN notion_error TEXT")
        if "obsidian_note_path" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN obsidian_note_path TEXT")
        if "dedupe_key" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN dedupe_key TEXT")
        if "usage" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN usage TEXT")
        if "interrupted" not in columns:
            await db.execute("ALTER TABLE jobs ADD COLUMN interrupted INTEGER NOT NULL DEFAULT 0")
        if "interruption_count" not in columns:
            await db.execute(
                "ALTER TABLE jobs ADD COLUMN interruption_count INTEGER NOT NULL DEFAULT 0"
            )
        if "processing_mode" not in columns:
            await db.execute(
                "ALTER TABLE jobs ADD COLUMN processing_mode TEXT NOT NULL DEFAULT 'cloud_public'"
            )
        if "external_processing_approved" not in columns:
            await db.execute(
                "ALTER TABLE jobs ADD COLUMN external_processing_approved INTEGER NOT NULL DEFAULT 0"
            )
        async with db.execute("SELECT job_id, url, processing_mode FROM jobs") as rows:
            for job_id, url, processing_mode in await rows.fetchall():
                base_dedupe_key, _ = submission_identity(str(url))
                dedupe_key = f"{base_dedupe_key}:{processing_mode or 'cloud_public'}"
                await db.execute(
                    "UPDATE jobs SET dedupe_key = ? WHERE job_id = ?",
                    (dedupe_key, job_id),
                )
        await db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_dedupe ON jobs(dedupe_key, status)")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS archive_entries (
                archive_id TEXT PRIMARY KEY,
                dedupe_key TEXT NOT NULL,
                processing_mode TEXT NOT NULL,
                obsidian_note_path TEXT,
                notion_page_id TEXT,
                result TEXT NOT NULL,
                summary TEXT NOT NULL,
                archived_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_dedupe ON archive_entries(dedupe_key)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_obsidian ON archive_entries(obsidian_note_path)"
        )
        async with db.execute(
            """SELECT * FROM jobs
               WHERE status = 'done' AND result IS NOT NULL AND summary IS NOT NULL
                 AND (obsidian_note_path IS NOT NULL OR notion_page_id IS NOT NULL)"""
        ) as completed_rows:
            for row in await completed_rows.fetchall():
                await _upsert_archive_row(db, _deserialize_job(row))
        if not settings.db_retain_transcript:
            async with db.execute(
                """SELECT job_id, result FROM jobs
                   WHERE status IN ('done', 'failed', 'cancelled') AND result IS NOT NULL"""
            ) as terminal_rows:
                for job_id, raw_result in await terminal_rows.fetchall():
                    result = TranscriptResult.model_validate_json(raw_result)
                    if result.transcript or result.segments:
                        await db.execute(
                            "UPDATE jobs SET result = ? WHERE job_id = ?",
                            (_without_transcript(result).model_dump_json(), job_id),
                        )
        await db.commit()
    path.chmod(0o600)


async def create_job(
    url: str,
    webhook_url: str | None = None,
    parent_job_id: str | None = None,
    dedupe_key: str | None = None,
    processing_mode: str = "nvidia_internal",
    external_processing_approved: bool = True,
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
        webhook_urls=[webhook_url] if webhook_url else [],
        parent_job_id=parent_job_id,
        dedupe_key=f"{dedupe_key or submission_identity(url)[0]}:{processing_mode}",
        processing_mode=processing_mode,
        external_processing_approved=external_processing_approved,
    )
    data = _serialize_job(job)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO jobs
                (job_id, url, status, stage, created_at, updated_at, retry_count,
                 result, summary, notion_page_id, notion_error, obsidian_note_path,
                 error, webhook_url, webhook_urls, parent_job_id, dedupe_key, usage,
                 interrupted, interruption_count, processing_mode,
                 external_processing_approved)
            VALUES
                (:job_id, :url, :status, :stage, :created_at, :updated_at, :retry_count,
                 :result, :summary, :notion_page_id, :notion_error, :obsidian_note_path,
                 :error, :webhook_url, :webhook_urls, :parent_job_id, :dedupe_key,
                 :usage, :interrupted, :interruption_count, :processing_mode,
                 :external_processing_approved)
            """,
            data,
        )
        await db.commit()
    logger.info("job_id=%s url=%.60s source=unknown event=job_created", job.job_id, url)
    return job


async def create_or_get_job(
    url: str,
    webhook_url: str | None = None,
    parent_job_id: str | None = None,
    *,
    processing_mode: str = "nvidia_internal",
    external_processing_approved: bool = False,
    db_path: str = DB_PATH,
) -> tuple[Job, bool]:
    """Atomically reuse an equivalent active/static-completed job or create one."""
    base_dedupe_key, reuse_completed = submission_identity(url)
    dedupe_key = f"{base_dedupe_key}:{processing_mode}"
    statuses = (
        ("pending", "processing", "done")
        if reuse_completed
        else (
            "pending",
            "processing",
        )
    )
    placeholders = ", ".join("?" for _ in statuses)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("BEGIN IMMEDIATE")
        async with db.execute(
            f"""SELECT * FROM jobs
                WHERE dedupe_key = ? AND status IN ({placeholders})
                ORDER BY created_at DESC LIMIT 1""",
            (dedupe_key, *statuses),
        ) as cursor:
            existing = await cursor.fetchone()
        if existing is not None:
            existing_job = _deserialize_job(existing)
            artifact_exists = existing_job.status != JobStatus.done or _obsidian_artifact_exists(
                existing_job.obsidian_note_path
            )
            if artifact_exists:
                if webhook_url and webhook_url not in existing_job.webhook_urls:
                    existing_job.webhook_urls.append(webhook_url)
                    if existing_job.webhook_url is None:
                        existing_job.webhook_url = webhook_url
                    await db.execute(
                        """UPDATE jobs
                           SET webhook_url = ?, webhook_urls = ?
                           WHERE job_id = ?""",
                        (
                            existing_job.webhook_url,
                            json.dumps(existing_job.webhook_urls),
                            existing_job.job_id,
                        ),
                    )
                await db.commit()
                return existing_job, False

        if reuse_completed:
            async with db.execute(
                """SELECT * FROM archive_entries
                   WHERE dedupe_key = ? ORDER BY updated_at DESC LIMIT 1""",
                (dedupe_key,),
            ) as cursor:
                archived = await cursor.fetchone()
            if archived is not None and _obsidian_artifact_exists(
                archived["obsidian_note_path"]
            ):
                now = _utcnow()
                job = Job(
                    job_id=str(uuid.uuid4()),
                    url=url,
                    status=JobStatus.done,
                    stage=JobStage.done,
                    created_at=now,
                    updated_at=now,
                    result=TranscriptResult.model_validate_json(archived["result"]),
                    summary=Summary.model_validate_json(archived["summary"]),
                    notion_page_id=archived["notion_page_id"],
                    obsidian_note_path=archived["obsidian_note_path"],
                    webhook_url=webhook_url,
                    webhook_urls=[webhook_url] if webhook_url else [],
                    parent_job_id=parent_job_id,
                    dedupe_key=dedupe_key,
                    processing_mode=processing_mode,
                    external_processing_approved=external_processing_approved,
                )
                data = _serialize_job(job)
                await db.execute(
                    """INSERT INTO jobs
                           (job_id, url, status, stage, created_at, updated_at, retry_count,
                            result, summary, notion_page_id, notion_error, obsidian_note_path,
                            error, webhook_url, webhook_urls, parent_job_id, dedupe_key, usage,
                            interrupted, interruption_count, processing_mode,
                            external_processing_approved)
                       VALUES
                           (:job_id, :url, :status, :stage, :created_at, :updated_at,
                            :retry_count, :result, :summary, :notion_page_id, :notion_error,
                            :obsidian_note_path, :error, :webhook_url, :webhook_urls,
                            :parent_job_id, :dedupe_key, :usage, :interrupted,
                            :interruption_count, :processing_mode,
                            :external_processing_approved)""",
                    data,
                )
                await db.commit()
                return job, False

        now = _utcnow()
        job = Job(
            job_id=str(uuid.uuid4()),
            url=url,
            status=JobStatus.pending,
            stage=JobStage.queued,
            created_at=now,
            updated_at=now,
            webhook_url=webhook_url,
            webhook_urls=[webhook_url] if webhook_url else [],
            parent_job_id=parent_job_id,
            dedupe_key=dedupe_key,
            processing_mode=processing_mode,
            external_processing_approved=external_processing_approved,
        )
        data = _serialize_job(job)
        await db.execute(
            """
            INSERT INTO jobs
                (job_id, url, status, stage, created_at, updated_at, retry_count,
                 result, summary, notion_page_id, notion_error, obsidian_note_path,
                 error, webhook_url, webhook_urls, parent_job_id, dedupe_key, usage,
                 interrupted, interruption_count, processing_mode,
                 external_processing_approved)
            VALUES
                (:job_id, :url, :status, :stage, :created_at, :updated_at, :retry_count,
                 :result, :summary, :notion_page_id, :notion_error, :obsidian_note_path,
                 :error, :webhook_url, :webhook_urls, :parent_job_id, :dedupe_key,
                 :usage, :interrupted, :interruption_count, :processing_mode,
                 :external_processing_approved)
            """,
            data,
        )
        await db.commit()
    logger.info("job_id=%s url=%.60s source=unknown event=job_created", job.job_id, url)
    return job, True


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


async def list_active_upload_urls(db_path: str = DB_PATH) -> set[str]:
    """Return upload references still owned by pending or processing jobs."""
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """SELECT url FROM jobs
               WHERE status IN ('pending', 'processing') AND url LIKE 'upload://%'"""
        ) as cursor:
            rows = await cursor.fetchall()
    return {str(row[0]) for row in rows}


async def find_notion_page_for_obsidian_note(
    obsidian_note_path: str,
    *,
    exclude_job_id: str,
    db_path: str = DB_PATH,
) -> str | None:
    """Return the prior Notion page paired with a canonical Obsidian note."""
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """SELECT notion_page_id FROM archive_entries
               WHERE obsidian_note_path = ? AND notion_page_id IS NOT NULL
               ORDER BY updated_at DESC LIMIT 1""",
            (obsidian_note_path,),
        ) as cursor:
            archived = await cursor.fetchone()
        if archived:
            return str(archived[0])
        async with db.execute(
            """SELECT notion_page_id FROM jobs
               WHERE obsidian_note_path = ?
                 AND notion_page_id IS NOT NULL
                 AND job_id != ?
               ORDER BY updated_at DESC LIMIT 1""",
            (obsidian_note_path, exclude_job_id),
        ) as cursor:
            row = await cursor.fetchone()
    return str(row[0]) if row else None


async def find_archived_output(
    obsidian_note_path: str,
    *,
    processing_mode: str,
    db_path: str = DB_PATH,
) -> tuple[Summary, str | None, str | None] | None:
    """Find a durable output checkpoint for a canonical media note."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT summary, obsidian_note_path, notion_page_id
               FROM archive_entries
               WHERE obsidian_note_path = ? AND processing_mode = ?
               ORDER BY updated_at DESC LIMIT 1""",
            (obsidian_note_path, processing_mode),
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    note_path = row["obsidian_note_path"]
    return (
        Summary.model_validate_json(row["summary"]),
        note_path if _obsidian_artifact_exists(note_path) else None,
        row["notion_page_id"],
    )


async def recover_incomplete_jobs(db_path: str = DB_PATH) -> list[str]:
    """Requeue interrupted jobs without consuming their provider retry budget."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("BEGIN IMMEDIATE")
        await db.execute(
            """UPDATE jobs
               SET status = CASE
                       WHEN result IS NOT NULL AND summary IS NOT NULL
                            AND (obsidian_note_path IS NOT NULL OR notion_page_id IS NOT NULL)
                           THEN 'pending'
                       WHEN interruption_count + 1 >= ? THEN 'failed'
                       ELSE 'pending'
                   END,
                   stage = CASE
                       WHEN result IS NOT NULL AND summary IS NOT NULL
                            AND (obsidian_note_path IS NOT NULL OR notion_page_id IS NOT NULL)
                           THEN stage
                       WHEN interruption_count + 1 >= ? THEN 'failed'
                       ELSE stage
                   END,
                   interruption_count = interruption_count + 1,
                   interrupted = 1,
                   error = CASE
                       WHEN NOT (
                                result IS NOT NULL AND summary IS NOT NULL
                                AND (obsidian_note_path IS NOT NULL OR notion_page_id IS NOT NULL)
                            )
                            AND interruption_count + 1 >= ?
                           THEN 'Job interrupted too many times during processing.'
                       ELSE error
                   END,
                   updated_at = ?
               WHERE status = 'processing'""",
            (
                MAX_INTERRUPTION_RECOVERIES,
                MAX_INTERRUPTION_RECOVERIES,
                MAX_INTERRUPTION_RECOVERIES,
                _utcnow().isoformat(),
            ),
        )
        async with db.execute(
            "SELECT job_id FROM jobs WHERE status = 'pending' ORDER BY created_at ASC"
        ) as cursor:
            job_ids = [str(row[0]) for row in await cursor.fetchall()]
        await db.commit()
    return job_ids


async def update_usage(
    job_id: str,
    usage: UsageStats,
    db_path: str = DB_PATH,
) -> None:
    """Persist provider usage without overwriting concurrent job state changes."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE jobs SET usage = ?, updated_at = ? WHERE job_id = ?",
            (usage.model_dump_json(), _utcnow().isoformat(), job_id),
        )
        await db.commit()


async def update_job(job: Job, db_path: str = DB_PATH) -> bool:
    """Persist a job unless doing so would resurrect a cancelled row."""
    job.updated_at = _utcnow()
    data = _serialize_job(job)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
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
                usage          = :usage,
                interrupted    = :interrupted,
                interruption_count = :interruption_count,
                error         = :error
            WHERE job_id = :job_id
              AND (status != 'cancelled' OR :status = 'cancelled')
            """,
            data,
        )
        await db.commit()
        return cursor.rowcount > 0


async def record_archive(job: Job, db_path: str = DB_PATH) -> None:
    """Persist output identity independently from expiring job history."""
    async with aiosqlite.connect(db_path) as db:
        await _upsert_archive_row(db, job)
        await db.commit()


async def redact_job_transcript(job_id: str, db_path: str = DB_PATH) -> None:
    """Remove transcript text and segments while preserving source metadata."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA secure_delete = ON")
        async with db.execute("SELECT result FROM jobs WHERE job_id = ?", (job_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None or row[0] is None:
            return
        result = TranscriptResult.model_validate_json(row[0])
        await db.execute(
            "UPDATE jobs SET result = ?, updated_at = ? WHERE job_id = ?",
            (_without_transcript(result).model_dump_json(), _utcnow().isoformat(), job_id),
        )
        await db.commit()


async def update_output_paths(
    job_id: str,
    *,
    obsidian_note_path: str | None,
    notion_page_id: str | None,
    notion_error: str | None,
    db_path: str = DB_PATH,
) -> None:
    """Record completed output side effects without changing job status."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """UPDATE jobs SET
                   obsidian_note_path = ?,
                   notion_page_id = ?,
                   notion_error = ?,
                   updated_at = ?
               WHERE job_id = ?""",
            (
                obsidian_note_path,
                notion_page_id,
                notion_error,
                _utcnow().isoformat(),
                job_id,
            ),
        )
        await db.commit()


async def mark_job_cancelled(job_id: str, db_path: str = DB_PATH) -> bool:
    """Atomically cancel only a job that has not already reached a final state."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """UPDATE jobs
               SET status = 'cancelled', stage = 'failed', updated_at = ?
               WHERE job_id = ? AND status IN ('pending', 'processing')""",
            (_utcnow().isoformat(), job_id),
        )
        await db.commit()
    return cursor.rowcount > 0


async def delete_job(job_id: str, db_path: str = DB_PATH) -> bool:
    """Delete a single job by ID. Returns True if a row was deleted."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA secure_delete = ON")
        cursor = await db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        await db.commit()
        return cursor.rowcount > 0


async def delete_jobs_by_status(status: str, db_path: str = DB_PATH) -> int:
    """Delete all jobs with the given status. Returns count of deleted rows."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA secure_delete = ON")
        cursor = await db.execute("DELETE FROM jobs WHERE status = ?", (status,))
        await db.commit()
        return cursor.rowcount


async def delete_old_jobs(max_age_days: int = 90, db_path: str = DB_PATH) -> int:
    """Delete jobs older than max_age_days. Returns count of deleted rows."""
    cutoff = (_utcnow() - timedelta(days=max_age_days)).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA secure_delete = ON")
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

"""Tests for the SQLite job queue.

Covers job creation, retrieval, state transitions, and retry counter.
All tests use a fresh temporary SQLite file — no shared state between tests.
"""


import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import job_queue
from models import JobStage, JobStatus, TranscriptResult


async def test_create_job_has_correct_defaults(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    assert job.job_id
    assert job.url == "https://youtube.com/watch?v=abc"
    assert job.status == JobStatus.pending
    assert job.retry_count == 0
    assert job.result is None
    assert job.summary is None
    assert job.error is None
    assert job.webhook_url is None


async def test_create_job_stores_webhook_url(db_path: str) -> None:
    job = await job_queue.create_job(
        "https://youtube.com/watch?v=abc", webhook_url="http://hook.example.com", db_path=db_path
    )
    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.webhook_url == "http://hook.example.com"


async def test_get_job_returns_none_for_missing_id(db_path: str) -> None:
    result = await job_queue.get_job("does-not-exist", db_path=db_path)
    assert result is None


async def test_get_job_round_trips_all_fields(db_path: str) -> None:
    created = await job_queue.create_job(
        "https://youtu.be/xyz", webhook_url="http://hook", db_path=db_path
    )
    fetched = await job_queue.get_job(created.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.job_id == created.job_id
    assert fetched.url == created.url
    assert fetched.webhook_url == "http://hook"
    assert fetched.status == JobStatus.pending


async def test_update_job_pending_to_processing(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.processing
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == JobStatus.processing


async def test_update_job_processing_to_done_with_result(db_path: str) -> None:
    from datetime import datetime

    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.done
    job.result = TranscriptResult(
        title="Test Video",
        source="youtube",
        url="https://youtube.com/watch?v=abc",
        channel_or_show="Test Channel",
        duration_seconds=300,
        transcript="This is the transcript.",
        published_at=datetime(2024, 1, 15, tzinfo=UTC),
        source_item_id="abc",
    )
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == JobStatus.done
    assert fetched.result is not None
    assert fetched.result.title == "Test Video"
    assert fetched.result.channel_or_show == "Test Channel"
    assert fetched.result.duration_seconds == 300
    assert fetched.result.transcript == "This is the transcript."
    assert fetched.result.source_item_id == "abc"


async def test_update_job_to_failed_stores_error(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.failed
    job.error = "No transcript available for this video."
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == JobStatus.failed
    assert fetched.error == "No transcript available for this video."


async def test_retry_count_persists(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.retry_count = 2
    job.status = JobStatus.failed
    job.error = "Transient error"
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.retry_count == 2


async def test_list_jobs_returns_newest_first(db_path: str) -> None:
    job1 = await job_queue.create_job("https://youtube.com/watch?v=aaa", db_path=db_path)
    job2 = await job_queue.create_job("https://youtube.com/watch?v=bbb", db_path=db_path)
    job3 = await job_queue.create_job("https://youtube.com/watch?v=ccc", db_path=db_path)

    jobs = await job_queue.list_jobs(db_path=db_path)
    assert jobs[0].job_id == job3.job_id
    assert jobs[1].job_id == job2.job_id
    assert jobs[2].job_id == job1.job_id


async def test_list_jobs_respects_limit(db_path: str) -> None:
    for i in range(5):
        await job_queue.create_job(f"https://youtube.com/watch?v={i}", db_path=db_path)

    jobs = await job_queue.list_jobs(limit=3, db_path=db_path)
    assert len(jobs) == 3


async def test_list_jobs_empty_db(db_path: str) -> None:
    jobs = await job_queue.list_jobs(db_path=db_path)
    assert jobs == []


async def test_updated_at_changes_on_update(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    original_updated_at = job.updated_at

    job.status = JobStatus.processing
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.updated_at >= original_updated_at


async def test_notion_page_id_persists(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.done
    job.notion_page_id = "abc-123-notion-page"
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.notion_page_id == "abc-123-notion-page"


async def test_output_metadata_persists(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.done
    job.obsidian_note_path = "Generated/Summaries/youtube-abc.md"
    job.notion_error = "Notion unavailable"
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.obsidian_note_path == "Generated/Summaries/youtube-abc.md"
    assert fetched.notion_error == "Notion unavailable"


async def test_init_db_migrates_existing_jobs_table(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    timestamp = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    with sqlite3.connect(path) as db:
        db.execute("""
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                result TEXT,
                summary TEXT,
                notion_page_id TEXT,
                error TEXT,
                webhook_url TEXT
            )
        """)
        db.execute(
            """INSERT INTO jobs
               (job_id, url, status, created_at, updated_at, retry_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("legacy-job", "https://youtu.be/abc", "pending", timestamp, timestamp, 0),
        )

    await job_queue.init_db(str(path))

    migrated = await job_queue.get_job("legacy-job", db_path=str(path))
    assert migrated is not None
    assert migrated.stage == JobStage.queued
    assert migrated.notion_error is None
    assert migrated.obsidian_note_path is None
    assert migrated.dedupe_key is not None


async def test_create_or_get_job_deduplicates_concurrent_static_urls(db_path: str) -> None:
    results = await asyncio.gather(
        job_queue.create_or_get_job(
            "https://youtube.com/watch?v=abc&t=30", db_path=db_path
        ),
        job_queue.create_or_get_job("https://youtu.be/abc?si=share", db_path=db_path),
    )

    assert results[0][0].job_id == results[1][0].job_id
    assert sorted(created for _, created in results) == [False, True]


async def test_create_or_get_job_reuses_completed_static_media(db_path: str) -> None:
    job, created = await job_queue.create_or_get_job(
        "https://youtube.com/watch?v=abc", db_path=db_path
    )
    assert created
    job.status = JobStatus.done
    await job_queue.update_job(job, db_path=db_path)

    duplicate, duplicate_created = await job_queue.create_or_get_job(
        "https://youtu.be/abc", db_path=db_path
    )

    assert not duplicate_created
    assert duplicate.job_id == job.job_id


async def test_create_or_get_job_refreshes_completed_feed(db_path: str) -> None:
    url = "https://feeds.example.com/show"
    job, _ = await job_queue.create_or_get_job(url, db_path=db_path)
    job.status = JobStatus.done
    await job_queue.update_job(job, db_path=db_path)

    refreshed, created = await job_queue.create_or_get_job(url, db_path=db_path)

    assert created
    assert refreshed.job_id != job.job_id


async def test_recover_incomplete_jobs_resets_processing_and_orders_pending(
    db_path: str,
) -> None:
    first = await job_queue.create_job("https://youtube.com/watch?v=first", db_path=db_path)
    second = await job_queue.create_job("https://youtube.com/watch?v=second", db_path=db_path)
    first.status = JobStatus.processing
    first.stage = JobStage.summarizing
    await job_queue.update_job(first, db_path=db_path)

    recovered = await job_queue.recover_incomplete_jobs(db_path=db_path)

    assert recovered == [first.job_id, second.job_id]
    reset = await job_queue.get_job(first.job_id, db_path=db_path)
    assert reset is not None
    assert reset.status == JobStatus.pending
    assert reset.stage == JobStage.queued


# ---------------------------------------------------------------------------
# New feature tests: stage, cancellation, deletion, TTL, parent_job_id
# ---------------------------------------------------------------------------


async def test_create_job_has_queued_stage(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    assert job.stage == JobStage.queued


async def test_stage_persists_through_update(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.stage = JobStage.transcribing
    job.status = JobStatus.processing
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.stage == JobStage.transcribing


async def test_cancelled_status(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.cancelled
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == JobStatus.cancelled


async def test_parent_job_id_persists(db_path: str) -> None:
    job = await job_queue.create_job(
        "https://youtube.com/watch?v=abc", parent_job_id="parent-123", db_path=db_path
    )
    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.parent_job_id == "parent-123"


async def test_delete_job(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    deleted = await job_queue.delete_job(job.job_id, db_path=db_path)
    assert deleted is True
    assert await job_queue.get_job(job.job_id, db_path=db_path) is None


async def test_delete_job_returns_false_for_missing(db_path: str) -> None:
    deleted = await job_queue.delete_job("nonexistent", db_path=db_path)
    assert deleted is False


async def test_delete_jobs_by_status(db_path: str) -> None:
    j1 = await job_queue.create_job("https://youtube.com/watch?v=a", db_path=db_path)
    j2 = await job_queue.create_job("https://youtube.com/watch?v=b", db_path=db_path)
    j3 = await job_queue.create_job("https://youtube.com/watch?v=c", db_path=db_path)
    j1.status = JobStatus.failed
    j1.error = "err"
    j2.status = JobStatus.failed
    j2.error = "err"
    j3.status = JobStatus.done
    await job_queue.update_job(j1, db_path=db_path)
    await job_queue.update_job(j2, db_path=db_path)
    await job_queue.update_job(j3, db_path=db_path)

    count = await job_queue.delete_jobs_by_status("failed", db_path=db_path)
    assert count == 2
    # done job should still exist
    assert await job_queue.get_job(j3.job_id, db_path=db_path) is not None


async def test_delete_old_jobs_respects_age(db_path: str) -> None:
    from datetime import timedelta

    # Create a job and manually backdate it
    job = await job_queue.create_job("https://youtube.com/watch?v=old", db_path=db_path)
    job.status = JobStatus.done
    job.created_at = job.created_at - timedelta(days=100)
    await job_queue.update_job(job, db_path=db_path)

    # Manually update created_at in the DB (update_job doesn't touch created_at)
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE jobs SET created_at = ? WHERE job_id = ?",
            (job.created_at.isoformat(), job.job_id),
        )
        await db.commit()

    deleted = await job_queue.delete_old_jobs(max_age_days=90, db_path=db_path)
    assert deleted == 1


async def test_delete_old_jobs_keeps_recent(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=new", db_path=db_path)
    job.status = JobStatus.done
    await job_queue.update_job(job, db_path=db_path)

    deleted = await job_queue.delete_old_jobs(max_age_days=90, db_path=db_path)
    assert deleted == 0

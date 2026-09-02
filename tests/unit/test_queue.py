"""Tests for the SQLite job queue.

Covers job creation, retrieval, state transitions, and retry counter.
All tests use a fresh temporary SQLite file — no shared state between tests.
"""

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import job_queue
from models import JobStage, JobStatus, Summary, TranscriptResult, UsageStats


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
    assert fetched.webhook_urls == ["http://hook.example.com"]


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
    job.usage = UsageStats(
        anthropic_requests=2,
        anthropic_input_tokens=1234,
        estimated_cost_usd=0.05,
    )
    await job_queue.update_job(job, db_path=db_path)

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.obsidian_note_path == "Generated/Summaries/youtube-abc.md"
    assert fetched.notion_error == "Notion unavailable"
    assert fetched.usage.anthropic_requests == 2
    assert fetched.usage.anthropic_input_tokens == 1234
    assert fetched.usage.estimated_cost_usd == 0.05


async def test_update_usage_does_not_overwrite_cancelled_status(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.cancelled
    await job_queue.update_job(job, db_path=db_path)

    await job_queue.update_usage(
        job.job_id,
        UsageStats(anthropic_requests=1, estimated_cost_usd=0.01),
        db_path=db_path,
    )

    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == JobStatus.cancelled
    assert fetched.usage.anthropic_requests == 1


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
    assert migrated.usage == UsageStats()


async def test_default_state_path_copies_legacy_database_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "repo" / "jobs.db"
    legacy.parent.mkdir()
    await job_queue.init_db(str(legacy))
    original = await job_queue.create_job(
        "https://youtube.com/watch?v=legacy", db_path=str(legacy)
    )
    destination = tmp_path / "Application Support" / "media-summarizer" / "jobs.db"
    monkeypatch.setattr(job_queue, "LEGACY_DB_PATH", legacy)
    monkeypatch.setattr(job_queue, "DB_PATH", str(destination))

    await job_queue.init_db(str(destination))

    migrated = await job_queue.get_job(original.job_id, db_path=str(destination))
    assert migrated is not None
    assert migrated.url == original.url
    assert legacy.is_file()
    assert destination.stat().st_mode & 0o777 == 0o600


async def test_create_or_get_job_deduplicates_concurrent_static_urls(db_path: str) -> None:
    results = await asyncio.gather(
        job_queue.create_or_get_job("https://youtube.com/watch?v=abc&t=30", db_path=db_path),
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


async def test_processing_boundary_persists_and_separates_deduplication(
    db_path: str,
) -> None:
    local, _ = await job_queue.create_or_get_job(
        "https://youtube.com/watch?v=abc",
        processing_mode="local",
        external_processing_approved=False,
        db_path=db_path,
    )
    cloud, cloud_created = await job_queue.create_or_get_job(
        "https://youtube.com/watch?v=abc",
        processing_mode="cloud_public",
        external_processing_approved=True,
        db_path=db_path,
    )
    fetched = await job_queue.get_job(local.job_id, db_path=db_path)

    assert fetched is not None
    assert fetched.processing_mode == "local"
    assert not fetched.external_processing_approved
    assert cloud_created
    assert cloud.job_id != local.job_id


async def test_finds_prior_notion_page_for_canonical_obsidian_note(db_path: str) -> None:
    prior = await job_queue.create_job("https://example.com/article", db_path=db_path)
    prior.status = JobStatus.done
    prior.obsidian_note_path = "Generated/Summaries/article-123.md"
    prior.notion_page_id = "notion-page-123"
    await job_queue.update_job(prior, db_path=db_path)

    found = await job_queue.find_notion_page_for_obsidian_note(
        prior.obsidian_note_path,
        exclude_job_id="new-job",
        db_path=db_path,
    )

    assert found == "notion-page-123"


async def test_finds_recorded_notion_page_from_cancelled_job(db_path: str) -> None:
    prior = await job_queue.create_job("https://example.com/article", db_path=db_path)
    prior.status = JobStatus.cancelled
    prior.obsidian_note_path = "Generated/Summaries/article-123.md"
    prior.notion_page_id = "notion-page-created-before-cancel"
    await job_queue.update_job(prior, db_path=db_path)

    found = await job_queue.find_notion_page_for_obsidian_note(
        prior.obsidian_note_path,
        exclude_job_id="new-job",
        db_path=db_path,
    )

    assert found == "notion-page-created-before-cancel"


async def test_create_or_get_job_refreshes_completed_feed(db_path: str) -> None:
    url = "https://feeds.example.com/show"
    job, _ = await job_queue.create_or_get_job(url, db_path=db_path)
    job.status = JobStatus.done
    await job_queue.update_job(job, db_path=db_path)

    refreshed, created = await job_queue.create_or_get_job(url, db_path=db_path)

    assert created
    assert refreshed.job_id != job.job_id


async def test_different_webhook_subscribes_to_existing_job(db_path: str) -> None:
    first, _ = await job_queue.create_or_get_job(
        "https://youtube.com/watch?v=abc",
        webhook_url="https://hooks.example.com/first",
        db_path=db_path,
    )

    second, created = await job_queue.create_or_get_job(
        "https://youtu.be/abc",
        webhook_url="https://hooks.example.com/second",
        db_path=db_path,
    )

    assert not created
    assert second.job_id == first.job_id
    assert second.webhook_urls == [
        "https://hooks.example.com/first",
        "https://hooks.example.com/second",
    ]


async def test_recover_incomplete_jobs_preserves_progress_and_orders_pending(
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
    assert reset.stage == JobStage.summarizing
    assert reset.retry_count == 0
    assert reset.interruption_count == 1
    assert reset.interrupted


async def test_recovery_fails_job_after_interruption_budget_is_exhausted(
    db_path: str,
) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.processing
    job.interruption_count = job_queue.MAX_INTERRUPTION_RECOVERIES - 1
    await job_queue.update_job(job, db_path=db_path)

    recovered = await job_queue.recover_incomplete_jobs(db_path=db_path)

    assert job.job_id not in recovered
    failed = await job_queue.get_job(job.job_id, db_path=db_path)
    assert failed is not None
    assert failed.status == JobStatus.failed
    assert failed.retry_count == 0
    assert failed.interruption_count == job_queue.MAX_INTERRUPTION_RECOVERIES


async def test_recovery_finalizes_checkpoint_even_after_interruption_budget(
    db_path: str,
) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.processing
    job.stage = JobStage.saving_notion
    job.interruption_count = job_queue.MAX_INTERRUPTION_RECOVERIES - 1
    job.result = TranscriptResult(
        title="Archived",
        source="youtube",
        url=job.url,
        channel_or_show="Channel",
        duration_seconds=60,
        transcript="durable transcript",
        source_item_id="abc",
    )
    job.summary = Summary(
        tldr="Archived summary",
        key_points=["Point"],
        tags=["test"],
        worth_rewatching=False,
    )
    job.obsidian_note_path = "Generated/Summaries/youtube-abc.md"
    await job_queue.update_job(job, db_path=db_path)

    recovered = await job_queue.recover_incomplete_jobs(db_path=db_path)

    assert recovered == [job.job_id]
    checkpoint = await job_queue.get_job(job.job_id, db_path=db_path)
    assert checkpoint is not None
    assert checkpoint.status == JobStatus.pending
    assert checkpoint.interruption_count == job_queue.MAX_INTERRUPTION_RECOVERIES


async def test_archive_identity_survives_job_deletion(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.done
    job.stage = JobStage.done
    job.result = TranscriptResult(
        title="Archived",
        source="youtube",
        url=job.url,
        channel_or_show="Channel",
        duration_seconds=60,
        transcript="do not retain this transcript",
        source_item_id="abc",
    )
    job.summary = Summary(
        tldr="Archived summary",
        key_points=["Point"],
        tags=["test"],
        worth_rewatching=False,
    )
    job.obsidian_note_path = "Generated/Summaries/youtube-abc.md"
    job.notion_page_id = "notion-abc"
    await job_queue.update_job(job, db_path=db_path)
    await job_queue.record_archive(job, db_path=db_path)
    await job_queue.delete_job(job.job_id, db_path=db_path)

    reused, created = await job_queue.create_or_get_job(
        "https://youtu.be/abc", db_path=db_path
    )

    assert not created
    assert reused.status == JobStatus.done
    assert reused.obsidian_note_path == job.obsidian_note_path
    assert reused.notion_page_id == job.notion_page_id
    assert reused.result is not None
    assert reused.result.transcript == ""


async def test_redact_job_transcript_preserves_metadata(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.result = TranscriptResult(
        title="Metadata",
        source="youtube",
        url=job.url,
        channel_or_show="Channel",
        duration_seconds=60,
        transcript="sensitive transcript",
        source_item_id="abc",
    )
    await job_queue.update_job(job, db_path=db_path)

    await job_queue.redact_job_transcript(job.job_id, db_path=db_path)

    redacted = await job_queue.get_job(job.job_id, db_path=db_path)
    assert redacted is not None and redacted.result is not None
    assert redacted.result.title == "Metadata"
    assert redacted.result.transcript == ""
    assert redacted.result.segments == []


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


async def test_stale_worker_update_cannot_resurrect_cancelled_job(db_path: str) -> None:
    stale = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    cancelled = stale.model_copy(deep=True)
    cancelled.status = JobStatus.cancelled
    await job_queue.update_job(cancelled, db_path=db_path)
    stale.status = JobStatus.processing
    stale.stage = JobStage.summarizing

    updated = await job_queue.update_job(stale, db_path=db_path)

    assert not updated
    fetched = await job_queue.get_job(stale.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == JobStatus.cancelled


async def test_atomic_cancel_does_not_overwrite_completed_job(db_path: str) -> None:
    job = await job_queue.create_job("https://youtube.com/watch?v=abc", db_path=db_path)
    job.status = JobStatus.done
    job.stage = JobStage.done
    await job_queue.update_job(job, db_path=db_path)

    cancelled = await job_queue.mark_job_cancelled(job.job_id, db_path=db_path)

    assert not cancelled
    fetched = await job_queue.get_job(job.job_id, db_path=db_path)
    assert fetched is not None
    assert fetched.status == JobStatus.done


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

"""Tests for the SQLite job queue.

Covers job creation, retrieval, state transitions, and retry counter.
All tests use a fresh temporary SQLite file — no shared state between tests.
"""


from datetime import UTC

import job_queue
from models import JobStatus, TranscriptResult


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

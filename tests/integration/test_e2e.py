"""End-to-end integration test — full pipeline from URL submission to Notion page."""

import pytest

import job_queue
from models import JobStatus
from pipeline import run_job

pytestmark = pytest.mark.integration

# "Me at the zoo" — first YouTube video, 19 seconds, stable captions
_SHORT_VIDEO_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


async def test_full_pipeline_youtube_to_notion(
    db_path: str,
    notion_test_db_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit a short YouTube URL and verify the full pipeline completes."""
    from config import settings
    monkeypatch.setattr(settings, "notion_database_id", notion_test_db_id)

    job = await job_queue.create_job(_SHORT_VIDEO_URL, db_path=db_path)
    await run_job(job.job_id, db_path=db_path)

    result = await job_queue.get_job(job.job_id, db_path=db_path)
    assert result is not None
    assert result.status == JobStatus.done, f"Job failed: {result.error}"


async def test_pipeline_result_has_transcript(
    db_path: str,
    notion_test_db_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config import settings
    monkeypatch.setattr(settings, "notion_database_id", notion_test_db_id)

    job = await job_queue.create_job(_SHORT_VIDEO_URL, db_path=db_path)
    await run_job(job.job_id, db_path=db_path)

    result = await job_queue.get_job(job.job_id, db_path=db_path)
    assert result is not None
    assert result.result is not None
    assert len(result.result.transcript) > 0


async def test_pipeline_result_has_summary(
    db_path: str,
    notion_test_db_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config import settings
    monkeypatch.setattr(settings, "notion_database_id", notion_test_db_id)

    job = await job_queue.create_job(_SHORT_VIDEO_URL, db_path=db_path)
    await run_job(job.job_id, db_path=db_path)

    result = await job_queue.get_job(job.job_id, db_path=db_path)
    assert result is not None
    assert result.summary is not None
    assert result.summary.tldr
    assert isinstance(result.summary.tags, list)


async def test_pipeline_creates_notion_page(
    db_path: str,
    notion_test_db_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config import settings
    monkeypatch.setattr(settings, "notion_database_id", notion_test_db_id)

    job = await job_queue.create_job(_SHORT_VIDEO_URL, db_path=db_path)
    await run_job(job.job_id, db_path=db_path)

    result = await job_queue.get_job(job.job_id, db_path=db_path)
    assert result is not None
    assert result.notion_page_id is not None
    assert len(result.notion_page_id.replace("-", "")) == 32


async def test_invalid_url_job_fails(db_path: str) -> None:
    job = await job_queue.create_job("https://open.spotify.com/episode/abc", db_path=db_path)
    await run_job(job.job_id, db_path=db_path)

    result = await job_queue.get_job(job.job_id, db_path=db_path)
    assert result is not None
    assert result.status == JobStatus.failed
    assert result.error is not None

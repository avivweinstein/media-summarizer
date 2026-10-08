from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from config import settings
from main import _job_to_response, cancel_job
from models import Job, JobStatus, TranscriptResult


def test_list_response_can_omit_transcript_without_mutating_job() -> None:
    job = Job(
        job_id="job-1",
        url="https://youtube.com/watch?v=abc",
        status=JobStatus.processing,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        result=TranscriptResult(
            title="Title",
            source="youtube",
            url="https://youtube.com/watch?v=abc",
            channel_or_show="Channel",
            duration_seconds=60,
            transcript="sensitive transcript",
        ),
    )

    response = _job_to_response(job, include_transcript=False)

    assert response.result is not None
    assert response.result.transcript == ""
    assert job.result is not None
    assert job.result.transcript == "sensitive transcript"


async def test_cancel_stops_active_work_and_redacts_transcript(mocker: MagicMock) -> None:
    job = Job(
        job_id="job-1",
        url="https://youtube.com/watch?v=abc",
        status=JobStatus.processing,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    mocker.patch("main.job_queue.get_job", new=AsyncMock(return_value=job))
    mocker.patch("main.job_queue.mark_job_cancelled", new=AsyncMock(return_value=True))
    redact = mocker.patch("main.job_queue.redact_job_transcript", new_callable=AsyncMock)
    cancel = mocker.patch("main.job_worker.cancel", new_callable=AsyncMock)
    mocker.patch.object(settings, "db_retain_transcript", False)

    result = await cancel_job(job.job_id)

    assert result == {"status": "cancelled"}
    cancel.assert_awaited_once_with(job.job_id)
    redact.assert_awaited_once_with(job.job_id)

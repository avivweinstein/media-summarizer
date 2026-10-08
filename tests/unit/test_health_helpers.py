from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from main import _create_and_enqueue, _obsidian_destinations_writable, _single_instance_lock
from models import Job, JobStatus


def test_single_instance_lock_rejects_second_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "service.lock"

    with _single_instance_lock(lock_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with _single_instance_lock(lock_path):
                pass


async def test_completed_duplicate_notifies_requesting_webhook(
    mocker: MagicMock,
) -> None:
    now = datetime.now(UTC)
    job = Job(
        job_id="existing",
        url="https://youtube.com/watch?v=abc",
        status=JobStatus.done,
        created_at=now,
        updated_at=now,
        webhook_url="https://hooks.example.com/callback",
    )
    mocker.patch("main.job_queue.create_or_get_job", return_value=(job, False))
    enqueue = mocker.patch("main.job_worker.enqueue")
    notify = mocker.patch("main._notify_webhook")

    result = await _create_and_enqueue(job.url, job.webhook_url)

    assert result is job
    enqueue.assert_not_called()
    notify.assert_awaited_once_with(
        job,
        urls=["https://hooks.example.com/callback"],
    )


def test_obsidian_destinations_writable_before_generated_dirs_exist(
    tmp_path: Path,
) -> None:
    assert _obsidian_destinations_writable(tmp_path, retain_transcript=True)


def test_obsidian_destinations_check_existing_generated_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_dir = tmp_path / "Generated" / "Summaries"
    transcript_dir = tmp_path / "Generated" / "Transcripts"
    summary_dir.mkdir(parents=True)
    transcript_dir.mkdir()
    monkeypatch.setattr(
        "main.os.access",
        lambda path, _mode: path != summary_dir,
    )

    assert not _obsidian_destinations_writable(tmp_path, retain_transcript=True)

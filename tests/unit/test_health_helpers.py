from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import settings
from main import (
    _create_and_enqueue,
    _obsidian_destinations_writable,
    _single_instance_lock,
    dashboard,
    health,
)
from models import Job, JobStatus


def test_single_instance_lock_rejects_second_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "service.lock"

    with _single_instance_lock(lock_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with _single_instance_lock(lock_path):
                pass


async def test_dashboard_labels_local_data_boundary(mocker: MagicMock) -> None:
    mocker.patch.object(settings, "processing_mode", "local")

    response = await dashboard()
    body = bytes(response.body).decode()

    assert "Local-only mode" in body
    assert "transcripts stay on this Mac" in body
    assert "__PROCESSING_MODE_LABEL__" not in body


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


async def test_local_health_never_checks_cloud_providers(tmp_path: Path, mocker: MagicMock) -> None:
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    model = tmp_path / "whisper.bin"
    model.write_bytes(b"model")
    mocker.patch.object(settings, "processing_mode", "local")
    mocker.patch.object(settings, "obsidian_vault_path", str(vault))
    mocker.patch.object(settings, "local_whisper_model", str(model))
    mocker.patch("main.shutil.which", return_value="/opt/homebrew/bin/whisper-cli")
    mocker.patch("main.job_queue.list_jobs", new=AsyncMock(return_value=[]))
    response = MagicMock()
    response.json.return_value = {"models": [{"name": settings.ollama_model}]}
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get.return_value = response
    mocker.patch("main.httpx.AsyncClient", return_value=client)

    result = await health(True)

    assert result["status"] == "ok"
    assert result["anthropic"] == "disabled (local mode)"
    assert result["openai"] == "disabled (local mode)"
    assert result["notion"] == "disabled (local mode)"

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from config import settings
from main import (
    _create_and_enqueue,
    _obsidian_destinations_writable,
    _require_processing_ready,
    _single_instance_lock,
    dashboard,
    health,
)
from models import Job, JobStatus
from nvidia_inference import NvidiaInferenceError


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


async def test_dashboard_labels_nvidia_boundary_and_removes_approval(
    mocker: MagicMock,
) -> None:
    mocker.patch.object(settings, "processing_mode", "nvidia_internal")

    response = await dashboard()
    body = bytes(response.body).decode()

    assert "NVIDIA internal processing" in body
    assert "summaries use NVIDIA Inference Hub" in body
    assert "Public or approved for external AI" not in body
    assert 'id="url-approved"' not in body
    assert 'id="upload-approved"' not in body
    assert "__URL_APPROVAL_CONTROL__" not in body


async def test_completed_duplicate_notifies_requesting_webhook(
    mocker: MagicMock,
) -> None:
    mocker.patch.object(settings, "processing_mode", "cloud_public")
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
    mocker.patch("transcriber.shutil.which", return_value="/opt/homebrew/bin/whisper-cli")
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


async def test_nvidia_health_checks_internal_model_and_local_whisper_only(
    tmp_path: Path, mocker: MagicMock
) -> None:
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    mocker.patch.object(settings, "processing_mode", "nvidia_internal")
    mocker.patch.object(settings, "obsidian_vault_path", str(vault))
    mocker.patch("main.job_queue.list_jobs", new=AsyncMock(return_value=[]))
    verify = mocker.patch("main.verify_nvidia_model_access", new=AsyncMock())
    whisper = mocker.patch("main.local_whisper_configuration")
    public_anthropic = mocker.patch("anthropic.AsyncAnthropic")
    public_openai = mocker.patch("openai.AsyncOpenAI")
    notion = mocker.patch("notion_client.AsyncClient")

    result = await health(True)

    assert result["status"] == "ok"
    assert result["nvidia_inference"] == "ok"
    assert result["local_whisper"] == "ok"
    assert result["anthropic"] == "disabled (NVIDIA internal mode)"
    assert result["openai"] == "disabled (NVIDIA internal mode)"
    assert result["notion"] == "disabled (nvidia_internal mode)"
    assert result["webhooks"] == "disabled (nvidia_internal mode)"
    verify.assert_awaited_once()
    whisper.assert_called_once()
    public_anthropic.assert_not_called()
    public_openai.assert_not_called()
    notion.assert_not_called()


async def test_nvidia_readiness_fails_closed_without_provider(
    mocker: MagicMock,
) -> None:
    mocker.patch.object(settings, "processing_mode", "nvidia_internal")
    mocker.patch("main.local_whisper_configuration")
    mocker.patch(
        "main.verify_nvidia_model_access",
        new=AsyncMock(side_effect=NvidiaInferenceError("NVIDIA endpoint unavailable.")),
    )

    with pytest.raises(HTTPException) as caught:
        await _require_processing_ready()

    assert caught.value.status_code == 503
    assert caught.value.detail == "NVIDIA endpoint unavailable."

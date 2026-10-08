"""Bounded live verification of NVIDIA-internal summarization."""

from pathlib import Path

import pytest

import job_queue
from config import settings
from models import JobStatus
from nvidia_inference import verify_nvidia_model_access
from pipeline import run_job
from sources.upload import upload_path

pytestmark = pytest.mark.integration


async def test_internal_text_upload_reaches_obsidian(
    tmp_path: Path,
    db_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not settings.nvidia_inference_api_key:
        pytest.skip("Integration test requires NVIDIA_INFERENCE_API_KEY")

    await verify_nvidia_model_access()
    vault = tmp_path / "Media-Library"
    (vault / ".obsidian").mkdir(parents=True)
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(settings, "processing_mode", "nvidia_internal")
    monkeypatch.setattr(settings, "obsidian_vault_path", str(vault))
    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "notion_enabled", True)
    monkeypatch.setattr(settings, "webhooks_enabled", True)

    url = "upload://6e379f72-dab2-474f-8820-bb5df64af085/internal-smoke.txt"
    original = upload_path(url)
    original.parent.mkdir(parents=True)
    original.write_text(
        "Media Summarizer uses NVIDIA Inference Hub and archives results locally."
    )
    job = await job_queue.create_job(
        url,
        processing_mode="nvidia_internal",
        external_processing_approved=False,
        db_path=db_path,
    )

    await run_job(job.job_id, db_path=db_path)

    completed = await job_queue.get_job(job.job_id, db_path=db_path)
    assert completed is not None
    assert completed.status == JobStatus.done, completed.error
    assert completed.processing_mode == "nvidia_internal"
    assert completed.result is not None
    assert completed.result.transcription_model == "local/text"
    assert completed.summary is not None
    assert completed.obsidian_note_path
    note = vault / completed.obsidian_note_path
    assert note.is_file()
    content = note.read_text()
    assert (
        'summary_model: "nvidia-inference/'
        "us/azure/anthropic/eccn-claude-sonnet-5\""
    ) in content
    assert "transcription_model: \"local/text\"" in content
    assert completed.notion_page_id is None
    assert not original.exists()

"""Credential-free integration coverage for the local knowledge pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import job_queue
from config import settings
from library import ask_library
from models import JobStatus, Summary
from pipeline import run_job
from sources.upload import upload_path

pytestmark = pytest.mark.integration


async def test_local_upload_archives_and_retrieves_without_cloud(
    tmp_path: Path,
    db_path: str,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MagicMock,
) -> None:
    vault = tmp_path / "Media-Library"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / "My-Notes").mkdir()
    (vault / "Attachments").mkdir()
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(settings, "obsidian_vault_path", str(vault))
    monkeypatch.setattr(settings, "processing_mode", "local")
    monkeypatch.setattr(settings, "obsidian_retain_transcript", True)
    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "notion_enabled", True)
    monkeypatch.setattr(settings, "webhooks_enabled", True)

    url = "upload://b6f10a58-7c75-4b1f-8ab0-03b25b0fa83c/context-notes.txt"
    original = upload_path(url)
    original.parent.mkdir(parents=True)
    original.write_text(
        "Context windows preserve recent information so a system can produce grounded answers."
    )
    local_summary = Summary(
        tldr="Context windows retain recent information for grounded answers.",
        key_points=["Context windows support grounded retrieval."],
        tags=["ai"],
        worth_rewatching=False,
    )
    response = MagicMock()
    response.json.return_value = {"message": {"content": local_summary.model_dump_json()}}
    response.raise_for_status.return_value = None
    ollama_client = AsyncMock()
    ollama_client.__aenter__.return_value = ollama_client
    ollama_client.__aexit__.return_value = None
    ollama_client.post.return_value = response
    local_provider = mocker.patch("summarizer.httpx.AsyncClient", return_value=ollama_client)
    notion = mocker.patch("pipeline.save_to_notion")
    job = await job_queue.create_job(
        url,
        processing_mode="local",
        external_processing_approved=False,
        db_path=db_path,
    )

    await run_job(job.job_id, db_path=db_path)

    completed = await job_queue.get_job(job.job_id, db_path=db_path)
    assert completed is not None
    assert completed.status == JobStatus.done
    assert completed.processing_mode == "local"
    assert completed.result is not None
    assert completed.result.transcript == ""
    assert completed.usage.local_summary_requests == 1
    assert completed.obsidian_note_path
    assert (vault / completed.obsidian_note_path).is_file()
    assert not original.exists()
    local_provider.assert_called_once()
    ollama_client.post.assert_awaited_once()
    assert ollama_client.post.call_args.args[0].endswith("/api/chat")
    notion.assert_not_called()

    answer = await ask_library(
        str(vault),
        "How do context windows support grounded retrieval?",
        limit=5,
        provider="extractive",
        ollama_model="unused",
        ollama_base_url="http://127.0.0.1:11434",
    )

    assert answer.citations
    assert answer.citations[0].note_path == completed.obsidian_note_path
    assert all(f"[{citation.index}]" in answer.answer for citation in answer.citations)

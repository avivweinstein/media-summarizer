from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from library import ask_library, search_library


def _vault(tmp_path: Path) -> Path:
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / "Generated" / "Summaries").mkdir(parents=True)
    (tmp_path / "Generated" / "Transcripts").mkdir()
    (tmp_path / "My-Notes").mkdir()
    return tmp_path


def _write_note(path: Path, title: str, body: str) -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                f'title: "{title}"',
                'media_id: "youtube-test"',
                'source_url: "https://youtube.com/watch?v=test"',
                "---",
                "",
                f"# {title}",
                "",
                body,
            ]
        )
    )


async def test_searches_generated_notes_with_stable_citations(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    _write_note(
        vault / "Generated" / "Summaries" / "youtube-test.md",
        "Zone 2 Training",
        "Aerobic base training improves endurance through steady cycling.",
    )

    hits = await search_library(str(vault), "aerobic cycling", limit=10)

    assert len(hits) == 1
    assert hits[0].title == "Zone 2 Training"
    assert hits[0].note_path == "Generated/Summaries/youtube-test.md"
    assert hits[0].source_url == "https://youtube.com/watch?v=test"
    assert hits[0].line_number > 0


async def test_search_does_not_read_personal_notes(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    _write_note(
        vault / "My-Notes" / "private.md",
        "Private",
        "confidential-project-codename",
    )

    hits = await search_library(str(vault), "confidential-project-codename")

    assert hits == []


async def test_extractive_answer_is_local_and_cited(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    _write_note(
        vault / "Generated" / "Summaries" / "youtube-test.md",
        "Recovery",
        "Persist checkpoints before calling an external provider.",
    )

    answer = await ask_library(
        str(vault),
        "How do checkpoints persist?",
        limit=5,
        provider="extractive",
        ollama_model="unused",
        ollama_base_url="http://127.0.0.1:11434",
    )

    assert answer.provider == "extractive"
    assert answer.answer.endswith("[1]")
    assert answer.citations[0].note_path == "Generated/Summaries/youtube-test.md"


async def test_ollama_receives_only_retrieved_generated_excerpt(
    tmp_path: Path,
    mocker: MagicMock,
) -> None:
    vault = _vault(tmp_path)
    _write_note(
        vault / "Generated" / "Summaries" / "youtube-test.md",
        "Local Retrieval",
        "Only retrieved excerpts are sent to the local model.",
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "A local answer [1]"}}
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_response
    client_factory = mocker.patch("library.httpx.AsyncClient", return_value=mock_client)

    answer = await ask_library(
        str(vault),
        "What is local retrieval?",
        limit=5,
        provider="ollama",
        ollama_model="local-model",
        ollama_base_url="http://127.0.0.1:11434",
    )

    assert answer.answer == "A local answer [1]"
    payload = mock_client.post.call_args.kwargs["json"]
    assert "Only retrieved excerpts" in payload["messages"][0]["content"]
    assert client_factory.call_args.kwargs["trust_env"] is False


async def test_ollama_answer_with_invalid_citation_falls_back_to_extractive(
    tmp_path: Path,
    mocker: MagicMock,
) -> None:
    vault = _vault(tmp_path)
    _write_note(
        vault / "Generated" / "Summaries" / "youtube-test.md",
        "Grounded Answers",
        "Grounded answers must cite retrieved material.",
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "Unsupported [99]"}}
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_response
    mocker.patch("library.httpx.AsyncClient", return_value=mock_client)

    answer = await ask_library(
        str(vault),
        "How are answers grounded?",
        limit=5,
        provider="ollama",
        ollama_model="local-model",
        ollama_base_url="http://127.0.0.1:11434",
    )

    assert answer.provider == "extractive"
    assert answer.answer.endswith("[1]")


async def test_search_ignores_stopwords_and_substring_matches(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    _write_note(
        vault / "Generated" / "Summaries" / "noise.md",
        "General Notes",
        "What is the thing and how does it work? The category is broad.",
    )
    _write_note(
        vault / "Generated" / "Summaries" / "relevant.md",
        "Checkpoint Recovery",
        "A checkpoint restores interrupted jobs after restart.",
    )

    hits = await search_library(str(vault), "How does checkpoint recovery work?", limit=1)

    assert hits[0].note_path == "Generated/Summaries/relevant.md"


async def test_ollama_rejects_nonlocal_endpoint(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    _write_note(
        vault / "Generated" / "Summaries" / "youtube-test.md",
        "Local Retrieval",
        "Only local inference may receive generated library excerpts.",
    )

    with pytest.raises(ValueError, match="local HTTP endpoint"):
        await ask_library(
            str(vault),
            "What receives library excerpts?",
            limit=5,
            provider="ollama",
            ollama_model="remote-model",
            ollama_base_url="https://models.example.com",
        )

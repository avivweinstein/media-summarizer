import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exceptions import ObsidianError
from models import Summary, TranscriptResult
from obsidian_writer import save_to_obsidian, source_id


def _result(**overrides: object) -> TranscriptResult:
    values: dict[str, object] = {
        "title": "Zone 2 Training: A Practical Guide",
        "source": "youtube",
        "url": "https://www.youtube.com/watch?v=abc123&t=30",
        "channel_or_show": "Example Creator",
        "duration_seconds": 3600,
        "thumbnail_url": "https://example.com/image.jpg",
        "transcript": "A complete transcript.",
        "published_at": datetime(2026, 8, 1, tzinfo=UTC),
        "transcription_model": "youtube/captions",
    }
    values.update(overrides)
    return TranscriptResult(**values)  # type: ignore[arg-type]


def _summary(**overrides: object) -> Summary:
    values: dict[str, object] = {
        "tldr": "A concise summary.",
        "key_points": ["First point.", "Second point."],
        "tags": ["fitness", "health"],
        "worth_rewatching": True,
    }
    values.update(overrides)
    return Summary(**values)  # type: ignore[arg-type]


def _vault(tmp_path: Path) -> Path:
    vault = tmp_path / "Media-Library"
    (vault / ".obsidian").mkdir(parents=True)
    return vault


def test_youtube_source_id_uses_video_id() -> None:
    standard = source_id(_result(url="https://youtube.com/watch?v=abc123&t=30"))
    short = source_id(_result(url="https://youtu.be/abc123?si=tracking"))

    assert standard == "youtube-abc123"
    assert short == standard


def test_podcast_source_id_uses_episode_identity() -> None:
    feed_url = "https://example.com/show.xml"
    first = source_id(
        _result(source="podcast", url=feed_url, source_item_id="episode-guid-1")
    )
    second = source_id(
        _result(source="podcast", url=feed_url, source_item_id="episode-guid-2")
    )

    assert first != second


async def test_save_creates_summary_and_transcript_with_metadata(tmp_path: Path) -> None:
    vault = _vault(tmp_path)

    relative_path = await save_to_obsidian(
        _result(),
        _summary(),
        str(vault),
        retain_transcript=True,
        summary_model="anthropic/test-model",
        added_at=datetime(2026, 9, 2, tzinfo=UTC),
    )

    summary_path = vault / relative_path
    transcript_path = next((vault / "Generated/Transcripts").glob("*.md"))
    content = summary_path.read_text()
    assert relative_path.startswith("Generated/Summaries/")
    assert 'media_id: "youtube-abc123"' in content
    assert 'source_url: "https://www.youtube.com/watch?v=abc123&t=30"' in content
    assert 'creator: "Example Creator"' in content
    assert "date_added: 2026-09-02" in content
    assert "published: 2026-08-01" in content
    assert "duration_seconds: 3600" in content
    assert 'summary_model: "anthropic/test-model"' in content
    assert 'transcription_model: "youtube/captions"' in content
    assert 'source_item_id: "youtube-abc123"' in content
    assert "key_points:\n  - \"First point.\"\n  - \"Second point.\"" in content
    assert "- First point." in content
    assert "worth_rewatching: true" in content
    assert "![[Generated/Transcripts/" in content
    assert "A complete transcript." in transcript_path.read_text()
    assert summary_path.stat().st_mode & 0o777 == 0o600
    assert transcript_path.stat().st_mode & 0o777 == 0o600


async def test_save_is_idempotent_and_preserves_existing_generated_note(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    first = await save_to_obsidian(
        _result(),
        _summary(),
        str(vault),
        retain_transcript=True,
        summary_model="anthropic/test-model",
    )
    summary_path = vault / first
    summary_path.write_text(summary_path.read_text() + "\nPersonal annotation.\n")

    second = await save_to_obsidian(
        _result(),
        _summary(tldr="Changed"),
        str(vault),
        retain_transcript=True,
        summary_model="anthropic/test-model",
    )

    assert second == first
    assert "Personal annotation." in summary_path.read_text()
    assert len(list((vault / "Generated/Summaries").glob("*.md"))) == 1
    assert len(list((vault / "Generated/Transcripts").glob("*.md"))) == 1


async def test_concurrent_saves_create_one_note(tmp_path: Path) -> None:
    vault = _vault(tmp_path)

    paths = await asyncio.gather(
        *[
            save_to_obsidian(
                _result(title=title),
                _summary(),
                str(vault),
                retain_transcript=True,
                summary_model="anthropic/test-model",
            )
            for title in ("Original Title", "Changed Metadata Title")
        ]
    )

    assert paths[0] == paths[1]
    assert len(list((vault / "Generated/Summaries").glob("*.md"))) == 1
    assert len(list((vault / "Generated/Transcripts").glob("*.md"))) == 1


async def test_long_unicode_title_does_not_affect_filename(tmp_path: Path) -> None:
    vault = _vault(tmp_path)

    relative_path = await save_to_obsidian(
        _result(title="測" * 500),
        _summary(),
        str(vault),
        retain_transcript=False,
        summary_model="anthropic/test-model",
    )

    assert Path(relative_path).name == "youtube-abc123.md"


async def test_transcript_retention_can_be_disabled(tmp_path: Path) -> None:
    vault = _vault(tmp_path)

    relative_path = await save_to_obsidian(
        _result(),
        _summary(tags=[]),
        str(vault),
        retain_transcript=False,
        summary_model="anthropic/test-model",
    )

    content = (vault / relative_path).read_text()
    assert "tags: []" in content
    assert "transcript_retained: false" in content
    assert not (vault / "Generated/Transcripts").exists()


async def test_invalid_vault_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ObsidianError, match="Not an Obsidian vault"):
        await save_to_obsidian(
            _result(),
            _summary(),
            str(tmp_path),
            retain_transcript=True,
            summary_model="anthropic/test-model",
        )


async def test_title_cannot_escape_generated_directory(tmp_path: Path) -> None:
    vault = _vault(tmp_path)

    relative_path = await save_to_obsidian(
        _result(title="../../escape"),
        _summary(),
        str(vault),
        retain_transcript=False,
        summary_model="anthropic/test-model",
    )

    assert relative_path.startswith("Generated/Summaries/")
    assert ".." not in Path(relative_path).name
    assert not (tmp_path / "escape--youtube-abc123.md").exists()

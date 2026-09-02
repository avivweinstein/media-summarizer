from pathlib import Path

import pytest

from main import _obsidian_destinations_writable


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

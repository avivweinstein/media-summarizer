import sqlite3
from pathlib import Path

import pytest

from scripts.backup_media_library import create_backup, restore_backup, verify_backup


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    database = tmp_path / "state" / "jobs.db"
    database.parent.mkdir()
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, status TEXT)")
        db.execute("INSERT INTO jobs VALUES ('job-1', 'done')")
    vault = tmp_path / "Media-Library"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / "Generated" / "Summaries").mkdir(parents=True)
    (vault / "Generated" / "Summaries" / "item.md").write_text("# Summary\n")
    return database, vault, tmp_path / "backups"


def test_backup_verifies_and_restores_to_new_destinations(tmp_path: Path) -> None:
    database, vault, destination = _fixture(tmp_path)

    snapshot = create_backup(database, vault, destination)
    verify_backup(snapshot)
    restored_db = tmp_path / "restore" / "jobs.db"
    restored_vault = tmp_path / "Restored-Library"
    restore_backup(snapshot, restored_db, restored_vault)

    with sqlite3.connect(restored_db) as db:
        assert db.execute("SELECT status FROM jobs WHERE job_id = 'job-1'").fetchone() == (
            "done",
        )
    assert (restored_vault / "Generated" / "Summaries" / "item.md").read_text() == (
        "# Summary\n"
    )


def test_backup_verification_detects_corruption(tmp_path: Path) -> None:
    database, vault, destination = _fixture(tmp_path)
    snapshot = create_backup(database, vault, destination)
    (snapshot / "vault" / "Generated" / "Summaries" / "item.md").write_text("corrupt")

    with pytest.raises(ValueError, match="checksum"):
        verify_backup(snapshot)


def test_backup_retention_removes_oldest_snapshot(tmp_path: Path) -> None:
    database, vault, destination = _fixture(tmp_path)
    first = create_backup(database, vault, destination, retain=1)
    second = create_backup(database, vault, destination, retain=1)

    assert not first.exists()
    assert second.exists()

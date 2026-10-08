import os
import sqlite3
from pathlib import Path

import pytest

from scripts import backup_media_library
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


def test_corrupt_backup_quarantine_is_bounded(tmp_path: Path) -> None:
    database, vault, destination = _fixture(tmp_path)
    for index in range(4):
        snapshot = create_backup(database, vault, destination, retain=1)
        if index < 3:
            (snapshot / "manifest.json").write_text("{}")

    assert len(list((destination / "quarantine").iterdir())) == 2
    assert len(list(destination.glob("media-library-*"))) == 1


def test_backup_rejects_vault_symlinks(tmp_path: Path) -> None:
    database, vault, destination = _fixture(tmp_path)
    external = tmp_path / "outside.md"
    external.write_text("not part of the vault")
    (vault / "linked.md").symlink_to(external)

    with pytest.raises(ValueError, match="symlinks"):
        create_backup(database, vault, destination)


def test_failed_restore_leaves_no_partial_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, vault, destination = _fixture(tmp_path)
    snapshot = create_backup(database, vault, destination)
    restored_db = tmp_path / "restore" / "jobs.db"
    restored_vault = tmp_path / "Restored-Library"

    def fail_copytree(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated copy failure")

    monkeypatch.setattr("scripts.backup_media_library.shutil.copytree", fail_copytree)
    with pytest.raises(OSError, match="simulated"):
        restore_backup(snapshot, restored_db, restored_vault)

    assert not restored_db.exists()
    assert not restored_vault.exists()


def test_restore_recovers_after_interruption_between_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, vault, destination = _fixture(tmp_path)
    snapshot = create_backup(database, vault, destination)
    restored_db = tmp_path / "restore" / "jobs.db"
    restored_vault = tmp_path / "Restored-Library"
    real_replace = os.replace

    def interrupt_before_vault(source: str | Path, target: str | Path) -> None:
        if Path(target) == restored_vault:
            raise KeyboardInterrupt
        real_replace(source, target)

    with monkeypatch.context() as context:
        context.setattr("scripts.backup_media_library.os.replace", interrupt_before_vault)
        with pytest.raises(KeyboardInterrupt):
            restore_backup(snapshot, restored_db, restored_vault)

    assert restored_db.exists()
    assert not restored_vault.exists()

    restore_backup(snapshot, restored_db, restored_vault)

    assert restored_db.exists()
    assert (restored_vault / "Generated" / "Summaries" / "item.md").is_file()


def test_restore_fsyncs_each_publication_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, vault, destination = _fixture(tmp_path)
    snapshot = create_backup(database, vault, destination)
    restored_db = tmp_path / "restore" / "jobs.db"
    restored_vault = tmp_path / "Restored-Library"
    calls: list[Path] = []
    monkeypatch.setattr("scripts.backup_media_library._fsync_restore_payload", lambda *_: None)
    monkeypatch.setattr(
        "scripts.backup_media_library._fsync_directory",
        lambda path: calls.append(path),
    )

    restore_backup(snapshot, restored_db, restored_vault)

    assert calls == [
        restored_db.parent,
        restored_db.parent,
        restored_vault.parent,
        restored_db.parent,
    ]


def test_restore_fsyncs_payload_before_publishing_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, vault, destination = _fixture(tmp_path)
    snapshot = create_backup(database, vault, destination)
    restored_db = tmp_path / "restore" / "jobs.db"
    restored_vault = tmp_path / "Restored-Library"
    events: list[str] = []

    real_payload_sync = backup_media_library._fsync_restore_payload
    real_marker_write = backup_media_library._write_restore_marker

    def sync_payload(database_path: Path, vault_path: Path) -> None:
        events.append("payload")
        real_payload_sync(database_path, vault_path)

    def write_marker(marker: Path, payload: dict[str, str]) -> None:
        events.append("marker")
        real_marker_write(marker, payload)

    monkeypatch.setattr(backup_media_library, "_fsync_restore_payload", sync_payload)
    monkeypatch.setattr(backup_media_library, "_write_restore_marker", write_marker)

    restore_backup(snapshot, restored_db, restored_vault)

    assert events == ["payload", "marker"]

"""Create, verify, and restore self-contained media-library snapshots."""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from config import settings
from job_queue import DB_PATH

SNAPSHOT_PREFIX = "media-library-"
BACKUP_FORMAT = "media-summarizer-backup-v1"
RESTORE_MARKER_FORMAT = "media-summarizer-restore-v1"
QUARANTINE_RETAIN = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root) != Path("manifest.json")
    }


def _reject_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Backup trees must not contain symlinks: {path}")


def verify_backup(snapshot: Path) -> None:
    _reject_symlinks(snapshot)
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Backup manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != BACKUP_FORMAT:
        raise ValueError(f"Unsupported backup format: {snapshot}")
    expected = manifest.get("sha256")
    if not isinstance(expected, dict) or expected != _file_hashes(snapshot):
        raise ValueError(f"Backup checksum verification failed: {snapshot}")
    with sqlite3.connect(snapshot / "jobs.db") as db:
        integrity = db.execute("PRAGMA integrity_check").fetchone()
    if integrity != ("ok",):
        raise ValueError(f"Backup database integrity check failed: {integrity}")


def create_backup(
    database: Path,
    vault: Path,
    destination: Path,
    *,
    retain: int = 14,
) -> Path:
    database = database.expanduser().resolve()
    vault = vault.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not database.is_file():
        raise ValueError(f"Job database does not exist: {database}")
    if not vault.is_dir() or not (vault / ".obsidian").is_dir():
        raise ValueError(f"Not an Obsidian vault: {vault}")
    if destination == vault or destination.is_relative_to(vault):
        raise ValueError("Backup destination cannot be inside the Obsidian vault.")
    if retain < 1:
        raise ValueError("Backup retention must be at least one snapshot.")
    _reject_symlinks(vault)

    destination_existed = destination.exists()
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not destination_existed:
        destination.chmod(0o700)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    snapshot = destination / f"{SNAPSHOT_PREFIX}{timestamp}"
    temporary = Path(tempfile.mkdtemp(prefix=".media-library-", dir=destination))
    try:
        backup_db = temporary / "jobs.db"
        with sqlite3.connect(database) as source, sqlite3.connect(backup_db) as target:
            source.backup(target)
        backup_db.chmod(0o600)
        shutil.copytree(vault, temporary / "vault")
        manifest = {
            "format": BACKUP_FORMAT,
            "created_at": datetime.now(UTC).isoformat(),
            "database_source": str(database),
            "vault_source": str(vault),
            "sha256": _file_hashes(temporary),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").chmod(0o600)
        verify_backup(temporary)
        os.replace(temporary, snapshot)
        snapshot.chmod(0o700)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    valid_snapshots: list[Path] = []
    corrupt_snapshots: list[Path] = []
    for path in destination.glob(f"{SNAPSHOT_PREFIX}*"):
        if not path.is_dir():
            continue
        try:
            verify_backup(path)
        except ValueError:
            corrupt_snapshots.append(path)
        else:
            valid_snapshots.append(path)
    for expired in sorted(valid_snapshots, reverse=True)[retain:]:
        shutil.rmtree(expired)
    if corrupt_snapshots:
        quarantine = destination / "quarantine"
        quarantine.mkdir(mode=0o700, exist_ok=True)
        quarantine.chmod(0o700)
        for corrupt in corrupt_snapshots:
            os.replace(corrupt, quarantine / corrupt.name)
        quarantined = sorted(
            (
                path
                for path in quarantine.iterdir()
                if path.is_dir() and path.name.startswith(SNAPSHOT_PREFIX)
            ),
            reverse=True,
        )
        for expired in quarantined[QUARANTINE_RETAIN:]:
            shutil.rmtree(expired)
    return snapshot


def _restore_marker_path(database_destination: Path) -> Path:
    return database_destination.parent / f".{database_destination.name}.restore.json"


def _write_restore_marker(marker: Path, payload: dict[str, str]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=marker.parent, delete=False
    ) as temporary:
        json.dump(payload, temporary, sort_keys=True)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        temporary_path.chmod(0o600)
        os.replace(temporary_path, marker)
    finally:
        temporary_path.unlink(missing_ok=True)


def _restored_content_matches(snapshot: Path, database: Path, vault: Path) -> bool:
    if not database.is_file() or not vault.is_dir():
        return False
    return _sha256(database) == _sha256(snapshot / "jobs.db") and _file_hashes(
        vault
    ) == _file_hashes(snapshot / "vault")


def _recover_interrupted_restore(
    marker: Path,
    snapshot: Path,
    database_destination: Path,
    vault_destination: Path,
) -> bool:
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "format": RESTORE_MARKER_FORMAT,
        "snapshot": str(snapshot),
        "database_destination": str(database_destination),
        "vault_destination": str(vault_destination),
    }
    if payload != expected:
        raise ValueError(f"Restore marker does not match this restore: {marker}")
    if _restored_content_matches(snapshot, database_destination, vault_destination):
        marker.unlink()
        return True
    database_destination.unlink(missing_ok=True)
    if vault_destination.exists():
        shutil.rmtree(vault_destination)
    marker.unlink()
    return False


def restore_backup(snapshot: Path, database_destination: Path, vault_destination: Path) -> None:
    snapshot = snapshot.expanduser().resolve()
    database_destination = database_destination.expanduser().resolve()
    vault_destination = vault_destination.expanduser().resolve()
    verify_backup(snapshot)
    database_destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    vault_destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker = _restore_marker_path(database_destination)
    if _recover_interrupted_restore(
        marker, snapshot, database_destination, vault_destination
    ):
        return
    if database_destination.exists() or vault_destination.exists():
        raise ValueError("Restore destinations must not already exist.")
    database_fd, database_stage_name = tempfile.mkstemp(
        prefix=".media-library-restore-",
        dir=database_destination.parent,
    )
    os.close(database_fd)
    database_stage = Path(database_stage_name)
    vault_stage = Path(
        tempfile.mkdtemp(prefix=".media-library-restore-", dir=vault_destination.parent)
    )
    vault_stage.rmdir()
    try:
        shutil.copy2(snapshot / "jobs.db", database_stage)
        database_stage.chmod(0o600)
        shutil.copytree(snapshot / "vault", vault_stage)
        vault_stage.chmod(0o700)
        with sqlite3.connect(database_stage) as db:
            integrity = db.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",) or not (vault_stage / ".obsidian").is_dir():
            raise ValueError("Staged backup restore failed validation.")
        _write_restore_marker(
            marker,
            {
                "format": RESTORE_MARKER_FORMAT,
                "snapshot": str(snapshot),
                "database_destination": str(database_destination),
                "vault_destination": str(vault_destination),
            },
        )
        os.replace(database_stage, database_destination)
        os.replace(vault_stage, vault_destination)
        marker.unlink()
    except Exception:
        database_destination.unlink(missing_ok=True)
        if vault_destination.exists():
            shutil.rmtree(vault_destination)
        marker.unlink(missing_ok=True)
        raise
    finally:
        database_stage.unlink(missing_ok=True)
        if vault_stage.exists():
            shutil.rmtree(vault_stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--database", default=DB_PATH)
    create.add_argument("--vault", default=settings.obsidian_vault_path)
    create.add_argument("--destination", default=settings.backup_dir)
    create.add_argument("--retain", type=int, default=14)

    verify = subparsers.add_parser("verify")
    verify.add_argument("snapshot")

    restore = subparsers.add_parser("restore")
    restore.add_argument("snapshot")
    restore.add_argument("--database-destination", required=True)
    restore.add_argument("--vault-destination", required=True)

    args = parser.parse_args()
    if args.action == "create":
        if not args.vault:
            raise SystemExit("Configure OBSIDIAN_VAULT_PATH or pass --vault.")
        snapshot = create_backup(
            Path(args.database),
            Path(args.vault),
            Path(args.destination),
            retain=args.retain,
        )
        print(snapshot)
    elif args.action == "verify":
        verify_backup(Path(args.snapshot))
        print("ok")
    else:
        restore_backup(
            Path(args.snapshot),
            Path(args.database_destination),
            Path(args.vault_destination),
        )


if __name__ == "__main__":
    main()

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
        if path.is_file() and not path.is_symlink() and path.name != "manifest.json"
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
    if manifest.get("format") != "media-summarizer-backup-v1":
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
            "format": "media-summarizer-backup-v1",
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

    snapshots = sorted(
        (
            path
            for path in destination.glob(f"{SNAPSHOT_PREFIX}*")
            if path.is_dir() and (path / "manifest.json").is_file()
        ),
        reverse=True,
    )
    for expired in snapshots[retain:]:
        try:
            verify_backup(expired)
        except ValueError:
            continue
        shutil.rmtree(expired)
    return snapshot


def restore_backup(snapshot: Path, database_destination: Path, vault_destination: Path) -> None:
    snapshot = snapshot.expanduser().resolve()
    database_destination = database_destination.expanduser().resolve()
    vault_destination = vault_destination.expanduser().resolve()
    verify_backup(snapshot)
    if database_destination.exists() or vault_destination.exists():
        raise ValueError("Restore destinations must not already exist.")
    database_destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    vault_destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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
        os.replace(database_stage, database_destination)
        os.replace(vault_stage, vault_destination)
    except Exception:
        database_destination.unlink(missing_ok=True)
        if vault_destination.exists():
            shutil.rmtree(vault_destination)
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

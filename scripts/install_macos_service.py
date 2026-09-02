"""Install or remove the Media Summarizer launchd user service."""

import argparse
import os
import plistlib
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

LABEL = "com.avivw.media-summarizer"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
BACKUP_LABEL = f"{LABEL}.backup"
BACKUP_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{BACKUP_LABEL}.plist"
LOG_DIR = Path.home() / "Library" / "Logs" / "media-summarizer"


def build_plist(
    project_dir: Path,
    host: str,
    port: int,
    obsidian_vault_path: Path | None = None,
    *,
    disable_notion: bool = False,
) -> dict[str, Any]:
    """Build the launchd configuration for a project checkout."""
    uvicorn = project_dir / ".venv" / "bin" / "uvicorn"
    environment = {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    }
    if obsidian_vault_path is not None:
        environment["OBSIDIAN_VAULT_PATH"] = str(obsidian_vault_path)
    if disable_notion:
        environment["NOTION_ENABLED"] = "false"

    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(uvicorn),
            "main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        "WorkingDirectory": str(project_dir),
        "EnvironmentVariables": environment,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Background",
        "StandardOutPath": str(LOG_DIR / "stdout.log"),
        "StandardErrorPath": str(LOG_DIR / "stderr.log"),
        "Umask": 0o077,
    }


def build_backup_plist(project_dir: Path, obsidian_vault_path: Path) -> dict[str, Any]:
    """Build a daily, local snapshot job for the database and Obsidian vault."""
    return {
        "Label": BACKUP_LABEL,
        "ProgramArguments": [
            str(project_dir / ".venv" / "bin" / "python"),
            "-m",
            "scripts.backup_media_library",
            "create",
        ],
        "WorkingDirectory": str(project_dir),
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "OBSIDIAN_VAULT_PATH": str(obsidian_vault_path),
        },
        "RunAtLoad": True,
        "StartInterval": 86_400,
        "ProcessType": "Background",
        "StandardOutPath": str(LOG_DIR / "backup.log"),
        "StandardErrorPath": str(LOG_DIR / "backup-error.log"),
        "Umask": 0o077,
    }
def _target(label: str = LABEL) -> str:
    return f"gui/{os.getuid()}/{label}"


def _wait_until_unloaded(timeout_seconds: float = 5.0, *, label: str = LABEL) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["launchctl", "print", _target(label)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return
        time.sleep(0.1)
    raise SystemExit(f"Timed out waiting for {_target(label)} to unload.")


def _write_plist(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        plistlib.dump(payload, temp)
        temp_path = Path(temp.name)
    temp_path.chmod(0o600)
    temp_path.replace(path)


def install(
    project_dir: Path,
    host: str,
    port: int,
    obsidian_vault_path: Path | None = None,
    *,
    disable_notion: bool = False,
) -> None:
    """Write, load, and start the launchd service."""
    project_dir = project_dir.resolve()
    uvicorn = project_dir / ".venv" / "bin" / "uvicorn"
    env_file = project_dir / ".env"
    if not uvicorn.is_file():
        raise SystemExit(f"Missing {uvicorn}; install the project environment first.")
    if not env_file.is_file():
        raise SystemExit(f"Missing {env_file}; configure API credentials first.")
    if obsidian_vault_path is not None:
        obsidian_vault_path = obsidian_vault_path.expanduser().resolve()
        if not (obsidian_vault_path / ".obsidian").is_dir():
            raise SystemExit(f"Not an Obsidian vault: {obsidian_vault_path}")

    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    LOG_DIR.chmod(0o700)
    payload = build_plist(
        project_dir,
        host,
        port,
        obsidian_vault_path,
        disable_notion=disable_notion,
    )
    _write_plist(PLIST_PATH, payload)

    subprocess.run(
        ["launchctl", "bootout", _target()],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_until_unloaded()
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootstrap", domain, str(PLIST_PATH)], check=True)
    subprocess.run(["launchctl", "enable", _target()], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", _target()], check=True)
    if obsidian_vault_path is not None:
        backup_payload = build_backup_plist(project_dir, obsidian_vault_path)
        _write_plist(BACKUP_PLIST_PATH, backup_payload)
        subprocess.run(
            ["launchctl", "bootout", _target(BACKUP_LABEL)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_until_unloaded(label=BACKUP_LABEL)
        subprocess.run(
            ["launchctl", "bootstrap", domain, str(BACKUP_PLIST_PATH)], check=True
        )
        subprocess.run(["launchctl", "enable", _target(BACKUP_LABEL)], check=True)
        subprocess.run(
            ["launchctl", "kickstart", "-k", _target(BACKUP_LABEL)], check=True
        )
    else:
        subprocess.run(
            ["launchctl", "bootout", _target(BACKUP_LABEL)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_until_unloaded(label=BACKUP_LABEL)
        BACKUP_PLIST_PATH.unlink(missing_ok=True)


def uninstall() -> None:
    """Stop the launchd service and remove its installed plist."""
    subprocess.run(
        ["launchctl", "bootout", _target()],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_until_unloaded()
    PLIST_PATH.unlink(missing_ok=True)
    subprocess.run(
        ["launchctl", "bootout", _target(BACKUP_LABEL)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_until_unloaded(label=BACKUP_LABEL)
    BACKUP_PLIST_PATH.unlink(missing_ok=True)


def status() -> None:
    """Print the launchd service status."""
    subprocess.run(["launchctl", "print", _target()], check=True)
    subprocess.run(["launchctl", "print", _target(BACKUP_LABEL)], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "uninstall", "status"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--obsidian-vault", default="")
    parser.add_argument(
        "--disable-notion",
        action="store_true",
        help="Disable optional Notion publishing for this service.",
    )
    args = parser.parse_args()

    if args.action == "install":
        obsidian_vault = Path(args.obsidian_vault) if args.obsidian_vault else None
        install(
            Path(__file__).resolve().parents[1],
            args.host,
            args.port,
            obsidian_vault,
            disable_notion=args.disable_notion,
        )
    elif args.action == "uninstall":
        uninstall()
    else:
        status()


if __name__ == "__main__":
    main()

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
LOG_DIR = Path.home() / "Library" / "Logs" / "media-summarizer"


def build_plist(project_dir: Path, host: str, port: int) -> dict[str, Any]:
    """Build the launchd configuration for a project checkout."""
    uvicorn = project_dir / ".venv" / "bin" / "uvicorn"
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
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        },
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Background",
        "StandardOutPath": str(LOG_DIR / "stdout.log"),
        "StandardErrorPath": str(LOG_DIR / "stderr.log"),
        "Umask": 0o077,
    }


def _target() -> str:
    return f"gui/{os.getuid()}/{LABEL}"


def _wait_until_unloaded(timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["launchctl", "print", _target()],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return
        time.sleep(0.1)
    raise SystemExit(f"Timed out waiting for {_target()} to unload.")


def install(project_dir: Path, host: str, port: int) -> None:
    """Write, load, and start the launchd service."""
    project_dir = project_dir.resolve()
    uvicorn = project_dir / ".venv" / "bin" / "uvicorn"
    env_file = project_dir / ".env"
    if not uvicorn.is_file():
        raise SystemExit(f"Missing {uvicorn}; install the project environment first.")
    if not env_file.is_file():
        raise SystemExit(f"Missing {env_file}; configure API credentials first.")

    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_plist(project_dir, host, port)
    with tempfile.NamedTemporaryFile(dir=PLIST_PATH.parent, delete=False) as temp:
        plistlib.dump(payload, temp)
        temp_path = Path(temp.name)
    temp_path.chmod(0o600)
    temp_path.replace(PLIST_PATH)

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


def status() -> None:
    """Print the launchd service status."""
    subprocess.run(["launchctl", "print", _target()], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "uninstall", "status"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.action == "install":
        install(Path(__file__).resolve().parents[1], args.host, args.port)
    elif args.action == "uninstall":
        uninstall()
    else:
        status()


if __name__ == "__main__":
    main()

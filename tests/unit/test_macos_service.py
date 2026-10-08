from pathlib import Path
from unittest.mock import MagicMock, call

from scripts.install_macos_service import (
    BACKUP_LABEL,
    LABEL,
    _wait_until_healthy,
    _wait_until_unloaded,
    build_backup_plist,
    build_plist,
    uninstall,
)


def test_build_plist_uses_localhost_and_project_environment() -> None:
    project_dir = Path("/Users/test/media-summarizer")

    plist = build_plist(project_dir, "127.0.0.1", 8000)

    assert plist["Label"] == LABEL
    assert plist["WorkingDirectory"] == str(project_dir)
    assert plist["ProgramArguments"] == [
        "/Users/test/media-summarizer/.venv/bin/uvicorn",
        "main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert str(plist["EnvironmentVariables"]["PATH"]).startswith("/opt/homebrew/bin:")
    assert plist["EnvironmentVariables"]["PROCESSING_MODE"] == "nvidia_internal"
    assert plist["EnvironmentVariables"]["NOTION_ENABLED"] == "false"
    assert plist["EnvironmentVariables"]["WEBHOOKS_ENABLED"] == "false"
    assert plist["Umask"] == 0o077


def test_build_plist_sets_obsidian_vault_environment() -> None:
    plist = build_plist(
        Path("/Users/test/media-summarizer"),
        "127.0.0.1",
        8000,
        Path("/Users/test/Documents/Media-Library"),
    )

    assert plist["EnvironmentVariables"]["OBSIDIAN_VAULT_PATH"] == (
        "/Users/test/Documents/Media-Library"
    )


def test_build_plist_can_disable_notion() -> None:
    plist = build_plist(
        Path("/Users/test/media-summarizer"),
        "127.0.0.1",
        8000,
        disable_notion=True,
    )

    assert plist["EnvironmentVariables"]["NOTION_ENABLED"] == "false"


def test_build_backup_plist_runs_daily_with_private_files() -> None:
    plist = build_backup_plist(
        Path("/Users/test/media-summarizer"),
        Path("/Users/test/Documents/Media-Library"),
    )

    assert plist["Label"] == BACKUP_LABEL
    assert plist["ProgramArguments"][-2:] == ["scripts.backup_media_library", "create"]
    assert plist["StartInterval"] == 86_400
    assert plist["RunAtLoad"] is True
    assert plist["Umask"] == 0o077


def test_wait_until_unloaded_retries_until_launchd_forgets_service(mocker: MagicMock) -> None:
    run = mocker.patch("scripts.install_macos_service.subprocess.run")
    run.side_effect = [MagicMock(returncode=0), MagicMock(returncode=1)]
    mocker.patch("scripts.install_macos_service.time.sleep")

    _wait_until_unloaded()

    assert run.call_count == 2


def test_wait_until_healthy_retries_until_service_is_ready(mocker: MagicMock) -> None:
    response = MagicMock()
    response.__enter__.return_value.status = 200
    request = mocker.patch(
        "scripts.install_macos_service.urlopen",
        side_effect=[OSError("not ready"), response],
    )
    mocker.patch("scripts.install_macos_service.time.sleep")

    _wait_until_healthy("127.0.0.1", 8000)

    assert request.call_count == 2


def test_uninstall_waits_for_service_before_removing_plist(
    mocker: MagicMock, tmp_path: Path
) -> None:
    plist_path = tmp_path / "service.plist"
    backup_plist_path = tmp_path / "backup.plist"
    plist_path.touch()
    backup_plist_path.touch()
    mocker.patch("scripts.install_macos_service.PLIST_PATH", plist_path)
    mocker.patch("scripts.install_macos_service.BACKUP_PLIST_PATH", backup_plist_path)
    mocker.patch("scripts.install_macos_service.subprocess.run")
    wait = mocker.patch("scripts.install_macos_service._wait_until_unloaded")

    uninstall()

    assert wait.call_args_list == [call(), call(label=BACKUP_LABEL)]
    assert not plist_path.exists()
    assert not backup_plist_path.exists()

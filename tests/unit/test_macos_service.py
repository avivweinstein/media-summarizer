from pathlib import Path

from scripts.install_macos_service import LABEL, build_plist


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
    assert plist["Umask"] == 0o077

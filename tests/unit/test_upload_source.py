from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import settings
from exceptions import UnsupportedURLError
from models import TranscriptionOutput
from sources.upload import UploadSource, cleanup_upload, reconcile_uploads, upload_path


async def test_text_upload_stays_until_pipeline_cleanup(tmp_path: Path, mocker: MagicMock) -> None:
    mocker.patch.object(settings, "upload_dir", str(tmp_path))
    url = "upload://8dcb5535-0de6-4833-beb8-1e0bc0d266d5/meeting-notes.txt"
    path = upload_path(url)
    path.write_text("Public notes with enough material to create a useful summary.")

    result = await UploadSource().fetch(url)

    assert result.source == "upload"
    assert result.title == "meeting-notes.txt"
    assert path.exists()
    cleanup_upload(url)
    assert not path.exists()


async def test_media_upload_uses_copy_for_crash_recovery(tmp_path: Path, mocker: MagicMock) -> None:
    mocker.patch.object(settings, "upload_dir", str(tmp_path))
    url = "upload://8dcb5535-0de6-4833-beb8-1e0bc0d266d5/interview.mp3"
    path = upload_path(url)
    path.write_bytes(b"audio")
    transcription = mocker.patch(
        "sources.upload.transcribe",
        new=AsyncMock(return_value=TranscriptionOutput(text="Interview transcript.")),
    )

    result = await UploadSource().fetch(url, job_id="job")

    assert result.transcript == "Interview transcript."
    assert path.exists()
    assert transcription.call_args.args[0] != path
    transcription.call_args.args[0].unlink(missing_ok=True)


def test_invalid_upload_reference_is_rejected(tmp_path: Path, mocker: MagicMock) -> None:
    mocker.patch.object(settings, "upload_dir", str(tmp_path))

    with pytest.raises(UnsupportedURLError, match="Invalid upload"):
        upload_path("upload://not-a-uuid/file.txt")


def test_reconcile_uploads_removes_only_orphans(tmp_path: Path, mocker: MagicMock) -> None:
    mocker.patch.object(settings, "upload_dir", str(tmp_path))
    active_url = "upload://8dcb5535-0de6-4833-beb8-1e0bc0d266d5/active.txt"
    orphan_url = "upload://107af971-ef44-4df6-bf47-c7abf75d753d/orphan.mp3"
    active = upload_path(active_url)
    orphan = upload_path(orphan_url)
    unrelated = tmp_path / "keep.txt"
    active.write_text("active")
    orphan.write_bytes(b"orphan")
    unrelated.write_text("not managed")

    removed = reconcile_uploads({active_url})

    assert removed == 1
    assert active.exists()
    assert not orphan.exists()
    assert unrelated.exists()

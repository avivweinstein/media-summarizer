from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, UploadFile

from config import settings
from main import _persist_upload, submit_upload
from sources.upload import cleanup_upload, upload_path


async def test_cloud_upload_requires_explicit_approval(
    tmp_path: Path, mocker: MagicMock
) -> None:
    mocker.patch.object(settings, "processing_mode", "cloud_public")
    mocker.patch.object(settings, "upload_dir", str(tmp_path))
    upload = UploadFile(filename="notes.txt", file=BytesIO(b"public notes"))

    with pytest.raises(HTTPException, match="explicit confirmation"):
        await submit_upload(upload, False, None)

    assert not list(tmp_path.iterdir())


async def test_persist_upload_uses_private_randomized_file(
    tmp_path: Path, mocker: MagicMock
) -> None:
    mocker.patch.object(settings, "upload_dir", str(tmp_path))
    upload = UploadFile(filename="notes.txt", file=BytesIO(b"public notes"))

    url = await _persist_upload(upload)
    path = upload_path(url)

    assert path.read_bytes() == b"public notes"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert "notes.txt" not in path.name
    cleanup_upload(url)

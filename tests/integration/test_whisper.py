"""Whisper integration tests — transcribe a real audio file locally."""

import shutil
from pathlib import Path

import pytest

from transcriber import TMP_DIR, ensure_tmp_dir, transcribe

pytestmark = pytest.mark.integration

_FIXTURE_MP3 = Path(__file__).parent.parent / "fixtures" / "short_test.mp3"


@pytest.fixture(autouse=True)
def setup_tmp_dir() -> None:
    ensure_tmp_dir()


async def test_transcribe_returns_timestamped_output() -> None:
    dest = TMP_DIR / "integration-whisper-test.mp3"
    shutil.copy(_FIXTURE_MP3, dest)

    result = await transcribe(
        dest,
        job_id="integration-test",
        processing_mode="nvidia_internal",
    )

    assert result.text
    assert result.segments


async def test_transcribe_cleans_up_file() -> None:
    dest = TMP_DIR / "integration-whisper-cleanup-test.mp3"
    shutil.copy(_FIXTURE_MP3, dest)

    await transcribe(
        dest,
        job_id="integration-test",
        processing_mode="nvidia_internal",
    )

    assert not dest.exists()


async def test_transcribe_missing_file_raises() -> None:
    from exceptions import TranscriptionError

    missing = TMP_DIR / "does-not-exist-integration.mp3"

    with pytest.raises(TranscriptionError, match="not found"):
        await transcribe(
            missing,
            job_id="integration-test",
            processing_mode="nvidia_internal",
        )

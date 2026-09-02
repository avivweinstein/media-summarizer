"""Tests for MP3 temp file cleanup.

Covers:
  - ensure_tmp_dir() creates the directory
  - ensure_tmp_dir() deletes leftover .mp3 files on startup
  - transcribe() deletes the file on success
  - transcribe() deletes the file even when transcription raises
"""

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import settings
from exceptions import TranscriptionError, UsageLimitError
from models import TranscriptionOutput, UsageStats
from transcriber import _communicate_with_timeout, ensure_tmp_dir, tmp_path_for_job, transcribe


class TestSubprocessCleanup:
    async def test_timeout_terminates_process(self) -> None:
        process = MagicMock()
        process.returncode = None
        process.communicate = AsyncMock(side_effect=TimeoutError)
        process.wait = AsyncMock(return_value=0)

        with pytest.raises(TranscriptionError, match="configured 1-second timeout"):
            await _communicate_with_timeout(process, 1, "Local Whisper")

        process.terminate.assert_called_once()
        process.wait.assert_awaited_once()

    async def test_cancellation_terminates_process(self) -> None:
        process = MagicMock()
        process.returncode = None
        process.communicate = AsyncMock(side_effect=asyncio.CancelledError)
        process.wait = AsyncMock(return_value=0)

        with pytest.raises(asyncio.CancelledError):
            await _communicate_with_timeout(process, 1, "Local Whisper")

        process.terminate.assert_called_once()


class TestEnsureTmpDir:
    def test_job_path_creates_missing_temp_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "missing" / "media-summarizer"

        with patch("transcriber.TMP_DIR", target):
            path = tmp_path_for_job("job")

        assert target.is_dir()
        assert target.stat().st_mode & 0o777 == 0o700
        assert path == target / "job.mp3"

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "media-summarizer"
        assert not target.exists()

        with patch("transcriber.TMP_DIR", target):
            ensure_tmp_dir()

        assert target.exists()
        assert target.stat().st_mode & 0o777 == 0o700

    def test_tightens_existing_directory_permissions(self, tmp_path: Path) -> None:
        target = tmp_path / "media-summarizer"
        target.mkdir(mode=0o755)

        with patch("transcriber.TMP_DIR", target):
            ensure_tmp_dir()

        assert target.stat().st_mode & 0o777 == 0o700

    def test_rejects_symlink_temp_directory(self, tmp_path: Path) -> None:
        actual = tmp_path / "actual"
        actual.mkdir()
        target = tmp_path / "media-summarizer"
        target.symlink_to(actual, target_is_directory=True)

        with patch("transcriber.TMP_DIR", target):
            with pytest.raises(TranscriptionError, match="not a symlink"):
                ensure_tmp_dir()

    def test_idempotent_when_dir_already_exists(self, tmp_path: Path) -> None:
        target = tmp_path / "media-summarizer"
        target.mkdir()

        with patch("transcriber.TMP_DIR", target):
            ensure_tmp_dir()  # Should not raise

        assert target.exists()

    def test_removes_leftover_mp3_files(self, tmp_path: Path) -> None:
        target = tmp_path / "media-summarizer"
        target.mkdir()
        leftover1 = target / "job-abc.mp3"
        leftover2 = target / "job-xyz.mp3"
        leftover1.write_bytes(b"fake audio 1")
        leftover2.write_bytes(b"fake audio 2")

        with patch("transcriber.TMP_DIR", target):
            ensure_tmp_dir()

        assert not leftover1.exists()
        assert not leftover2.exists()

    def test_preserves_non_mp3_files(self, tmp_path: Path) -> None:
        target = tmp_path / "media-summarizer"
        target.mkdir()
        other_file = target / "notes.txt"
        other_file.write_text("keep me")

        with patch("transcriber.TMP_DIR", target):
            ensure_tmp_dir()

        assert other_file.exists()

    def test_empty_dir_does_not_raise(self, tmp_path: Path) -> None:
        target = tmp_path / "media-summarizer"
        target.mkdir()

        with patch("transcriber.TMP_DIR", target):
            ensure_tmp_dir()  # Should not raise


class TestTranscribeFileCleanup:
    @pytest.fixture(autouse=True)
    def _mock_audio_duration(self, mocker: MagicMock) -> None:
        mocker.patch("transcriber._probe_audio_duration", return_value=120)

    async def test_tracks_request_duration_and_cost(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mp3 = tmp_path / "usage.mp3"
        mp3.write_bytes(b"x" * 100)
        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create.return_value = "Transcript text."
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)
        usage = UsageStats()

        await transcribe(
            mp3,
            duration_seconds=120,
            usage=usage,
            processing_mode="cloud_public",
        )

        assert usage.openai_requests == 1
        assert usage.openai_audio_seconds == 120
        assert usage.estimated_cost_usd == pytest.approx(0.012)

    async def test_local_mode_never_calls_openai(self, tmp_path: Path, mocker: MagicMock) -> None:
        media = tmp_path / "local.mp3"
        media.write_bytes(b"audio")
        local = mocker.patch(
            "transcriber._transcribe_local",
            new=AsyncMock(return_value=TranscriptionOutput(text="Local transcript.")),
        )
        openai = mocker.patch("transcriber.AsyncOpenAI")
        result = await transcribe(media, job_id="local-job", processing_mode="local")

        assert result.text == "Local transcript."
        local.assert_awaited_once()
        openai.assert_not_called()

    async def test_nvidia_internal_mode_never_calls_openai(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        media = tmp_path / "internal.mp3"
        media.write_bytes(b"audio")
        local = mocker.patch(
            "transcriber._transcribe_local",
            new=AsyncMock(return_value=TranscriptionOutput(text="Internal transcript.")),
        )
        openai = mocker.patch("transcriber.AsyncOpenAI")

        result = await transcribe(
            media,
            job_id="internal-job",
            processing_mode="nvidia_internal",
        )

        assert result.text == "Internal transcript."
        local.assert_awaited_once()
        openai.assert_not_called()

    async def test_persists_reservation_before_openai_call(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mp3 = tmp_path / "durable.mp3"
        mp3.write_bytes(b"x" * 100)
        events: list[str] = []
        mock_client = AsyncMock()

        async def api_call(**_kwargs: object) -> str:
            events.append("api")
            return "Transcript text."

        async def persist(_usage: UsageStats) -> None:
            events.append("persist")

        mock_client.audio.transcriptions.create.side_effect = api_call
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)

        await transcribe(
            mp3,
            duration_seconds=60,
            persist_usage=persist,
            processing_mode="cloud_public",
        )

        assert events == ["persist", "api"]

    async def test_duration_limit_blocks_before_api_call(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mp3 = tmp_path / "too-long.mp3"
        mp3.write_bytes(b"x" * 100)
        mock_client = AsyncMock()
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)
        mocker.patch.object(settings, "max_audio_duration_seconds", 60)

        with pytest.raises(UsageLimitError, match="duration"):
            await transcribe(mp3, duration_seconds=120, processing_mode="cloud_public")

        mock_client.audio.transcriptions.create.assert_not_called()

    async def test_unknown_duration_fails_closed(self, tmp_path: Path, mocker: MagicMock) -> None:
        mp3 = tmp_path / "unknown-duration.mp3"
        mp3.write_bytes(b"x" * 100)
        mocker.patch("transcriber._probe_audio_duration", return_value=0)
        mock_client = AsyncMock()
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)

        with pytest.raises(TranscriptionError, match="determine audio duration"):
            await transcribe(mp3, duration_seconds=30, processing_mode="cloud_public")

        mock_client.audio.transcriptions.create.assert_not_called()

    async def test_existing_request_usage_counts_toward_limit(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mp3 = tmp_path / "request-limit.mp3"
        mp3.write_bytes(b"x" * 100)
        mock_client = AsyncMock()
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)
        usage = UsageStats(openai_requests=settings.max_openai_requests_per_job)

        with pytest.raises(UsageLimitError, match="request limit"):
            await transcribe(
                mp3,
                duration_seconds=60,
                usage=usage,
                processing_mode="cloud_public",
            )

        mock_client.audio.transcriptions.create.assert_not_called()

    async def test_deletes_file_on_success(self, tmp_path: Path, mocker: MagicMock) -> None:
        mp3 = tmp_path / "test-job.mp3"
        mp3.write_bytes(b"x" * 100)

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create.return_value = "Transcript text."
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)

        await transcribe(mp3, processing_mode="cloud_public")

        assert not mp3.exists()

    async def test_deletes_file_on_api_failure(self, tmp_path: Path, mocker: MagicMock) -> None:
        import httpx
        from openai import RateLimitError

        mp3 = tmp_path / "test-job.mp3"
        mp3.write_bytes(b"x" * 100)

        mock_client = AsyncMock()
        mock_resp = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com"))
        mock_client.audio.transcriptions.create.side_effect = RateLimitError(
            message="rate limited", response=cast(Any, mock_resp), body=None
        )
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)

        with pytest.raises(TranscriptionError):
            await transcribe(mp3, processing_mode="cloud_public")

        assert not mp3.exists()

    async def test_deletes_file_on_unexpected_exception(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mp3 = tmp_path / "test-job.mp3"
        mp3.write_bytes(b"x" * 100)

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create.side_effect = RuntimeError("unexpected!")
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)

        # RuntimeError bubbles up as-is (not wrapped) since it's not an API error
        with pytest.raises(Exception):
            await transcribe(mp3, processing_mode="cloud_public")

        assert not mp3.exists()

    async def test_file_too_large_without_ffmpeg_raises_before_api_call(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        from transcriber import WHISPER_MAX_BYTES

        mp3 = tmp_path / "big.mp3"
        mp3.write_bytes(b"x" * (WHISPER_MAX_BYTES + 1))

        mock_client = AsyncMock()
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)
        mocker.patch(
            "transcriber._compress_for_whisper",
            side_effect=TranscriptionError("ffmpeg not installed"),
        )

        with pytest.raises(TranscriptionError, match="ffmpeg"):
            await transcribe(mp3, processing_mode="cloud_public")

        # API should never have been called
        mock_client.audio.transcriptions.create.assert_not_called()
        # File still cleaned up
        assert not mp3.exists()

    async def test_file_too_large_compresses_and_transcribes(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        from transcriber import WHISPER_MAX_BYTES

        mp3 = tmp_path / "big.mp3"
        mp3.write_bytes(b"x" * (WHISPER_MAX_BYTES + 1))

        async def fake_compress(src: Path, dst: Path) -> None:
            dst.write_bytes(b"x" * 100)  # "compressed" file well under 25 MB

        mocker.patch("transcriber._compress_for_whisper", side_effect=fake_compress)

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create.return_value = "Compressed transcript."
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)

        result = await transcribe(mp3, processing_mode="cloud_public")

        assert result.text == "Compressed transcript."
        mock_client.audio.transcriptions.create.assert_called_once()
        # Both original and compressed files cleaned up
        assert not mp3.exists()
        assert not (tmp_path / "big_compressed.mp3").exists()

    async def test_preserves_whisper_segment_timestamps(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        mp3 = tmp_path / "timestamped.mp3"
        mp3.write_bytes(b"x" * 100)
        response = MagicMock()
        response.text = "First thought. Second thought."
        response.segments = [
            {"start": 0.0, "end": 4.2, "text": "First thought."},
            {"start": 4.2, "end": 8.0, "text": "Second thought."},
        ]
        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create.return_value = response
        mocker.patch("transcriber.AsyncOpenAI", return_value=mock_client)

        result = await transcribe(mp3, processing_mode="cloud_public")

        assert result.segments[1].start_seconds == 4.2
        call = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call["response_format"] == "verbose_json"
        assert call["timestamp_granularities"] == ["segment"]

    async def test_missing_file_raises_transcription_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.mp3"

        with pytest.raises(TranscriptionError, match="not found"):
            await transcribe(missing, processing_mode="cloud_public")

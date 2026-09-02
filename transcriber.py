"""OpenAI Whisper API wrapper.

Downloads MP3 to /tmp/media-summarizer/{job_id}.mp3, transcribes, deletes.
Temp file is always deleted — even if transcription fails.
Large files (>25 MB) are re-encoded to 32 kbps mono via ffmpeg before upload.
"""

import asyncio
import json
import logging
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from config import settings
from exceptions import TranscriptionError, UsageLimitError
from models import TranscriptionOutput, TranscriptSegment, UsageStats

logger = logging.getLogger(__name__)

TMP_DIR = Path("/tmp/media-summarizer")

# Whisper API hard limit
WHISPER_MAX_BYTES = 25 * 1024 * 1024  # 25 MB


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


async def _communicate_with_timeout(
    process: asyncio.subprocess.Process,
    timeout_seconds: int,
    label: str,
) -> tuple[bytes, bytes]:
    try:
        return await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.CancelledError:
        await asyncio.shield(_stop_process(process))
        raise
    except TimeoutError as error:
        await _stop_process(process)
        raise TranscriptionError(
            f"{label} exceeded its configured {timeout_seconds}-second timeout."
        ) from error


def ensure_tmp_dir() -> None:
    """Create the temp dir and delete any leftover MP3s from crashed jobs."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    temporary_suffixes = {".json", ".m4a", ".mov", ".mp3", ".mp4", ".wav", ".webm"}
    for file in TMP_DIR.iterdir():
        if file.is_file() and file.suffix.casefold() in temporary_suffixes:
            file.unlink(missing_ok=True)
            removed += 1
    if removed:
        logger.warning("job_id=- url=- source=- event=startup_cleanup removed_files=%d", removed)


def tmp_path_for_job(job_id: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    return TMP_DIR / f"{job_id}.mp3"


async def _compress_for_whisper(src: Path, dst: Path) -> None:
    """Re-encode audio to 32 kbps mono using ffmpeg.

    Raises TranscriptionError if ffmpeg is unavailable or fails.
    Typical reduction: 128 kbps stereo → 32 kbps mono = ~8x smaller.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",  # mono
        "-ab",
        "32k",  # 32 kbps — sufficient for speech
        "-ar",
        "16000",  # 16 kHz sample rate
        "-f",
        "mp3",
        str(dst),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await _communicate_with_timeout(
            proc, settings.local_ffmpeg_timeout_seconds, "ffmpeg conversion"
        )
        if proc.returncode != 0:
            raise TranscriptionError(f"ffmpeg compression failed: {stderr.decode()[:300]}")
    except FileNotFoundError:
        raise TranscriptionError(
            "Audio file exceeds Whisper's 25 MB limit and ffmpeg is not installed. "
            "Install ffmpeg (sudo apt install ffmpeg) or use a shorter episode."
        )


async def convert_to_mp3(src: Path, dst: Path) -> None:
    """Normalize supported audio or video input to speech-optimized MP3."""
    await _compress_for_whisper(src, dst)


async def _probe_audio_duration(path: Path) -> float:
    """Return media duration from ffprobe, or zero when it cannot be determined."""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await _communicate_with_timeout(
            process, min(settings.local_ffmpeg_timeout_seconds, 60), "ffprobe"
        )
        if process.returncode == 0:
            return max(0.0, float(stdout.decode().strip()))
    except (FileNotFoundError, ValueError):
        pass
    return 0.0


def transcription_model_name(processing_mode: str) -> str:
    if processing_mode == "local":
        return "local/whisper.cpp"
    return "openai/whisper-1"


async def _transcribe_local(path: Path, job_id: str) -> TranscriptionOutput:
    executable = shutil.which(settings.local_whisper_executable)
    model = Path(settings.local_whisper_model).expanduser()
    if executable is None or not model.is_file():
        raise TranscriptionError(
            "Local mode requires LOCAL_WHISPER_EXECUTABLE and a valid LOCAL_WHISPER_MODEL."
        )
    wav_path = TMP_DIR / f"{job_id}-local.wav"
    output_prefix = TMP_DIR / f"{job_id}-local"
    json_path = output_prefix.with_suffix(".json")
    try:
        ffmpeg = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, ffmpeg_error = await _communicate_with_timeout(
            ffmpeg, settings.local_ffmpeg_timeout_seconds, "Local audio conversion"
        )
        if ffmpeg.returncode != 0:
            raise TranscriptionError(
                f"Local audio conversion failed: {ffmpeg_error.decode()[:300]}"
            )
        process = await asyncio.create_subprocess_exec(
            executable,
            "-m",
            str(model),
            "-f",
            str(wav_path),
            "-oj",
            "-of",
            str(output_prefix),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await _communicate_with_timeout(
            process, settings.local_whisper_timeout_seconds, "Local Whisper"
        )
        if process.returncode != 0 or not json_path.is_file():
            raise TranscriptionError(f"Local Whisper failed: {stderr.decode()[:300]}")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        raw_segments = data.get("transcription") or data.get("segments") or []
        segments: list[TranscriptSegment] = []
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            offsets = item.get("offsets", {})
            start = item.get("start", 0)
            end = item.get("end", 0)
            if isinstance(offsets, dict) and "from" in offsets and "to" in offsets:
                start = float(offsets.get("from", 0)) / 1000
                end = float(offsets.get("to", 0)) / 1000
            text = str(item.get("text", "")).strip()
            if text:
                segments.append(
                    TranscriptSegment(start_seconds=float(start), end_seconds=float(end), text=text)
                )
        transcript = " ".join(segment.text for segment in segments)
        if not transcript:
            transcript = str(data.get("text", "")).strip()
        if not transcript:
            raise TranscriptionError("Local Whisper returned an empty transcript.")
        return TranscriptionOutput(text=transcript, segments=segments)
    except (OSError, json.JSONDecodeError) as error:
        raise TranscriptionError(f"Local Whisper output could not be read: {error}") from error
    finally:
        wav_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)


async def transcribe(
    mp3_path: Path,
    job_id: str = "-",
    *,
    duration_seconds: float | None = None,
    usage: UsageStats | None = None,
    persist_usage: Callable[[UsageStats], Awaitable[None]] | None = None,
    processing_mode: str = "cloud_public",
) -> TranscriptionOutput:
    """Send an MP3 file to Whisper and return text with segment timestamps.

    Always deletes the file on exit, regardless of success or failure.
    Raises TranscriptionError on any API or file failure.
    """
    log = f"job_id={job_id} url=- source=podcast"
    compressed_path: Path | None = None
    usage_tracker = usage or UsageStats()

    try:
        if not mp3_path.exists():
            raise TranscriptionError(f"MP3 file not found: {mp3_path}")

        file_size = mp3_path.stat().st_size
        if file_size > settings.max_audio_download_bytes:
            raise UsageLimitError(
                f"Audio download is {file_size / 1e6:.0f} MB, exceeding the configured "
                f"{settings.max_audio_download_bytes / 1e6:.0f} MB per-job limit."
            )
        logger.info(
            "%s event=transcribe_start path=%s size_mb=%.1f", log, mp3_path.name, file_size / 1e6
        )

        audio_seconds = await _probe_audio_duration(mp3_path)
        if audio_seconds <= 0:
            raise TranscriptionError(
                "Could not determine audio duration; refusing an unbounded Whisper request."
            )
        if (
            duration_seconds is not None
            and duration_seconds > 0
            and abs(duration_seconds - audio_seconds) > 60
        ):
            logger.warning(
                "%s event=duration_metadata_mismatch published=%s probed=%.1f",
                log,
                duration_seconds,
                audio_seconds,
            )
        if audio_seconds > settings.max_audio_duration_seconds:
            raise UsageLimitError(
                f"Audio duration is {audio_seconds / 3600:.1f} hours, exceeding the "
                f"configured {settings.max_audio_duration_seconds / 3600:.1f}-hour limit."
            )
        if processing_mode == "local":
            result = await _transcribe_local(mp3_path, job_id)
            logger.info("%s event=transcribe_done chars=%d provider=local", log, len(result.text))
            return result
        estimated_cost = audio_seconds / 60 * settings.whisper_cost_per_minute_usd
        if usage_tracker.estimated_cost_usd + estimated_cost > settings.max_estimated_cost_usd:
            raise UsageLimitError(
                f"Estimated transcription cost ${estimated_cost:.2f} exceeds the "
                f"configured ${settings.max_estimated_cost_usd:.2f} per-job limit."
            )
        if usage_tracker.openai_requests >= settings.max_openai_requests_per_job:
            raise UsageLimitError("Configured OpenAI per-job request limit reached.")

        if file_size > WHISPER_MAX_BYTES:
            logger.info(
                "%s event=compress_start size_mb=%.1f reason=exceeds_whisper_limit",
                log,
                file_size / 1e6,
            )
            compressed_path = mp3_path.parent / f"{mp3_path.stem}_compressed.mp3"
            await _compress_for_whisper(mp3_path, compressed_path)
            upload_path = compressed_path
            file_size = upload_path.stat().st_size
            logger.info("%s event=compress_done size_mb=%.1f", log, file_size / 1e6)
            if file_size > WHISPER_MAX_BYTES:
                raise TranscriptionError(
                    f"Audio file is {file_size / 1e6:.0f} MB even after ffmpeg compression "
                    "(exceeds Whisper's 25 MB limit). The episode may be too long."
                )
        else:
            upload_path = mp3_path

        client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)

        usage_tracker.openai_requests += 1
        usage_tracker.openai_audio_seconds += audio_seconds
        usage_tracker.estimated_cost_usd = round(
            usage_tracker.estimated_cost_usd + estimated_cost,
            6,
        )
        if persist_usage is not None:
            await persist_usage(usage_tracker)

        with open(upload_path, "rb") as f:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        raw_text = response if isinstance(response, str) else getattr(response, "text", "")
        transcript = str(raw_text)
        raw_segments = [] if isinstance(response, str) else getattr(response, "segments", [])
        segments = [
            TranscriptSegment(
                start_seconds=float(item.get("start", 0) if isinstance(item, dict) else item.start),
                end_seconds=float(item.get("end", 0) if isinstance(item, dict) else item.end),
                text=str(item.get("text", "") if isinstance(item, dict) else item.text),
            )
            for item in (raw_segments or [])
            if (item.get("text", "") if isinstance(item, dict) else item.text).strip()
        ]

        logger.info("%s event=transcribe_done chars=%d", log, len(transcript))
        return TranscriptionOutput(text=transcript, segments=segments)

    except TranscriptionError:
        raise
    except RateLimitError as e:
        msg = str(e).lower()
        if any(w in msg for w in ("quota", "credit", "billing", "insufficient")):
            raise TranscriptionError(
                "OpenAI API quota exceeded. Please check your billing at platform.openai.com."
            ) from e
        raise TranscriptionError("Whisper API rate limit reached. Please try again shortly.") from e
    except AuthenticationError as e:
        raise TranscriptionError(
            "Invalid OpenAI API key. Please check OPENAI_API_KEY in .env."
        ) from e
    except BadRequestError as e:
        raise TranscriptionError(f"Whisper rejected the audio file: {e}") from e
    except APIConnectionError as e:
        raise TranscriptionError(f"Could not connect to OpenAI API: {e}") from e
    except APIStatusError as e:
        raise TranscriptionError(f"OpenAI API error (HTTP {e.status_code}): {e.message}") from e
    except OSError as e:
        raise TranscriptionError(f"Failed to read audio file: {e}") from e
    finally:
        mp3_path.unlink(missing_ok=True)
        if compressed_path is not None:
            compressed_path.unlink(missing_ok=True)
        logger.info("%s event=mp3_deleted path=%s", log, mp3_path.name)

"""OpenAI Whisper API wrapper.

Downloads MP3 to /tmp/media-summarizer/{job_id}.mp3, transcribes, deletes.
Temp file is always deleted — even if transcription fails.
Large files (>25 MB) are re-encoded to 32 kbps mono via ffmpeg before upload.
"""

import asyncio
import logging
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
from exceptions import TranscriptionError

logger = logging.getLogger(__name__)

TMP_DIR = Path("/tmp/media-summarizer")

# Whisper API hard limit
WHISPER_MAX_BYTES = 25 * 1024 * 1024  # 25 MB


def ensure_tmp_dir() -> None:
    """Create the temp dir and delete any leftover MP3s from crashed jobs."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for f in TMP_DIR.glob("*.mp3"):
        f.unlink(missing_ok=True)
        removed += 1
    if removed:
        logger.warning(
            "job_id=- url=- source=- event=startup_cleanup removed_files=%d", removed
        )


def tmp_path_for_job(job_id: str) -> Path:
    return TMP_DIR / f"{job_id}.mp3"


async def _compress_for_whisper(src: Path, dst: Path) -> None:
    """Re-encode audio to 32 kbps mono using ffmpeg.

    Raises TranscriptionError if ffmpeg is unavailable or fails.
    Typical reduction: 128 kbps stereo → 32 kbps mono = ~8x smaller.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",        # mono
        "-ab", "32k",      # 32 kbps — sufficient for speech
        "-ar", "16000",    # 16 kHz sample rate
        "-f", "mp3",
        str(dst),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise TranscriptionError(
                f"ffmpeg compression failed: {stderr.decode()[:300]}"
            )
    except FileNotFoundError:
        raise TranscriptionError(
            "Audio file exceeds Whisper's 25 MB limit and ffmpeg is not installed. "
            "Install ffmpeg (sudo apt install ffmpeg) or use a shorter episode."
        )


async def transcribe(mp3_path: Path, job_id: str = "-") -> str:
    """Send an MP3 file to Whisper and return the transcript text.

    Always deletes the file on exit, regardless of success or failure.
    Raises TranscriptionError on any API or file failure.
    """
    log = f"job_id={job_id} url=- source=podcast"
    compressed_path: Path | None = None

    try:
        if not mp3_path.exists():
            raise TranscriptionError(f"MP3 file not found: {mp3_path}")

        file_size = mp3_path.stat().st_size
        logger.info("%s event=transcribe_start path=%s size_mb=%.1f", log, mp3_path.name, file_size / 1e6)

        if file_size > WHISPER_MAX_BYTES:
            logger.info(
                "%s event=compress_start size_mb=%.1f reason=exceeds_whisper_limit",
                log, file_size / 1e6,
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

        client = AsyncOpenAI(api_key=settings.openai_api_key)

        with open(upload_path, "rb") as f:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
            )

        # response_format="text" returns a str directly
        transcript = response if isinstance(response, str) else str(response)

        logger.info("%s event=transcribe_done chars=%d", log, len(transcript))
        return transcript

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
        raise TranscriptionError("Invalid OpenAI API key. Please check OPENAI_API_KEY in .env.") from e
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

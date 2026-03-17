"""OpenAI Whisper API wrapper.

Downloads MP3 to /tmp/media-summarizer/{job_id}.mp3, transcribes, deletes.
Temp file is always deleted — even if transcription fails.
"""

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


async def transcribe(mp3_path: Path, job_id: str = "-") -> str:
    """Send an MP3 file to Whisper and return the transcript text.

    Always deletes the file on exit, regardless of success or failure.
    Raises TranscriptionError on any API or file failure.
    """
    log = f"job_id={job_id} url=- source=podcast"

    try:
        if not mp3_path.exists():
            raise TranscriptionError(f"MP3 file not found: {mp3_path}")

        file_size = mp3_path.stat().st_size
        logger.info("%s event=transcribe_start path=%s size_mb=%.1f", log, mp3_path.name, file_size / 1e6)

        if file_size > WHISPER_MAX_BYTES:
            raise TranscriptionError(
                f"Audio file is {file_size / 1e6:.0f} MB, which exceeds Whisper's 25 MB limit. "
                "Try a shorter episode or a direct MP3 URL with lower bitrate."
            )

        client = AsyncOpenAI(api_key=settings.openai_api_key)

        with open(mp3_path, "rb") as f:
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
        logger.info("%s event=mp3_deleted path=%s", log, mp3_path.name)

"""Crash-safe local upload ingestion."""

import shutil
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

from config import settings
from exceptions import UnsupportedURLError
from models import TranscriptResult, UsageStats
from transcriber import tmp_path_for_job, transcribe, transcription_model_name

TEXT_EXTENSIONS = {".md", ".txt"}
MEDIA_EXTENSIONS = {".m4a", ".mp3", ".mp4", ".mov", ".wav", ".webm"}
UPLOAD_EXTENSIONS = TEXT_EXTENSIONS | MEDIA_EXTENSIONS


def upload_path(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "upload" or not parsed.hostname:
        raise UnsupportedURLError("Invalid upload reference.")
    try:
        upload_id = str(uuid.UUID(parsed.hostname))
    except ValueError as error:
        raise UnsupportedURLError("Invalid upload reference.") from error
    suffix = Path(unquote(parsed.path)).suffix.casefold()
    if suffix not in UPLOAD_EXTENSIONS:
        raise UnsupportedURLError("Unsupported upload type.")
    root = Path(settings.upload_dir).expanduser().resolve()
    path = (root / f"{upload_id}{suffix}").resolve()
    if not path.is_relative_to(root):
        raise UnsupportedURLError("Invalid upload path.")
    return path


def cleanup_upload(url: str) -> None:
    if urlparse(url).scheme == "upload":
        try:
            upload_path(url).unlink(missing_ok=True)
        except UnsupportedURLError:
            return


class UploadSource:
    async def fetch(
        self,
        url: str,
        job_id: str = "-",
        *,
        usage: UsageStats | None = None,
        persist_usage: Callable[[UsageStats], Awaitable[None]] | None = None,
    ) -> TranscriptResult:
        path = upload_path(url)
        if not path.is_file():
            raise UnsupportedURLError("Uploaded file is missing.")
        original_name = Path(unquote(urlparse(url).path)).name
        suffix = path.suffix.casefold()
        if suffix in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) < 20:
                raise UnsupportedURLError("Uploaded text is too short to summarize.")
            if len(text) > settings.max_transcript_chars:
                raise UnsupportedURLError("Uploaded text exceeds the transcript-size limit.")
            return TranscriptResult(
                title=original_name,
                source="upload",
                url=url,
                channel_or_show="",
                duration_seconds=0,
                transcript=text,
                transcription_model="local/text",
                source_item_id=urlparse(url).hostname,
            )

        transcription_path = tmp_path_for_job(job_id).with_suffix(suffix)
        shutil.copyfile(path, transcription_path)
        transcription = await transcribe(
            transcription_path,
            job_id,
            usage=usage,
            persist_usage=persist_usage,
        )
        return TranscriptResult(
            title=original_name,
            source="upload",
            url=url,
            channel_or_show="",
            duration_seconds=0,
            transcript=transcription.text,
            segments=transcription.segments,
            transcription_model=transcription_model_name(),
            source_item_id=urlparse(url).hostname,
        )

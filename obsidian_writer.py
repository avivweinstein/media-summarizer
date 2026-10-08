"""Write durable, idempotent Markdown records into an Obsidian vault."""

import asyncio
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

from exceptions import ObsidianError
from models import Summary, TranscriptResult, UsageStats

SUMMARY_DIR = Path("Generated/Summaries")
TRANSCRIPT_DIR = Path("Generated/Transcripts")


def source_id(result: TranscriptResult) -> str:
    """Return a stable, filename-safe identifier for a media item."""
    parsed = urlparse(result.url.strip())
    hostname = (parsed.hostname or "").lower().removeprefix("www.")
    if result.source == "youtube":
        video_id = result.source_item_id or ""
        if hostname == "youtu.be":
            video_id = video_id or parsed.path.strip("/").split("/")[0]
        elif not video_id:
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        safe_video_id = re.sub(r"[^A-Za-z0-9_-]", "", video_id)
        if safe_video_id:
            return f"youtube-{safe_video_id}"

    if result.source_item_id:
        identity = f"{result.source}:{result.source_item_id}"
    else:
        normalized = urlunparse(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", parsed.query, "")
        )
        identity = normalized
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    safe_source = re.sub(r"[^a-z0-9]+", "-", result.source.lower()).strip("-") or "media"
    return f"{safe_source}-{digest}"


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _format_timestamp(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _moment_link(result: TranscriptResult, seconds: int) -> str | None:
    if result.source != "youtube":
        return None
    parsed = urlparse(result.url)
    query = [(key, value) for key, value in parse_qsl(parsed.query) if key != "t"]
    query.append(("t", f"{seconds}s"))
    return urlunparse(parsed._replace(query=urlencode(query)))


def _atomic_write_once(path: Path, content: str) -> None:
    """Create a file atomically without replacing an existing note."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temp:
            temp.write(content)
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = Path(temp.name)
        temp_path.chmod(0o600)
        try:
            os.link(temp_path, path)
        except FileExistsError:
            pass
    except OSError as error:
        raise ObsidianError(f"Failed to write Obsidian note {path}: {error}") from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _find_existing(directory: Path, media_id: str) -> Path | None:
    matches = sorted(directory.glob(f"*--{media_id}.md"))
    return matches[0] if matches else None


def _render_transcript(result: TranscriptResult, media_id: str) -> str:
    transcript_lines = (
        [
            f"**{_format_timestamp(segment.start_seconds)}** {segment.text.strip()}"
            for segment in result.segments
        ]
        if result.segments
        else [result.transcript.strip()]
    )
    lines = [
        "---",
        "type: media-transcript",
        f"media_id: {_yaml_string(media_id)}",
        f"title: {_yaml_string(result.title)}",
        f"source: {_yaml_string(result.source)}",
        f"source_url: {_yaml_string(result.url)}",
        f"source_item_id: {_yaml_string(result.source_item_id or media_id)}",
        f"transcription_model: {_yaml_string(result.transcription_model or 'unknown')}",
        "---",
        "",
        f"# Transcript: {result.title}",
        "",
        *transcript_lines,
        "",
    ]
    return "\n".join(lines)


def _render_summary(
    result: TranscriptResult,
    summary: Summary,
    media_id: str,
    added_at: datetime,
    summary_model: str,
    transcript_path: Path | None,
    usage: UsageStats,
) -> str:
    published = result.published_at.date().isoformat() if result.published_at else ""
    tag_lines = ["tags:", *[f"  - {_yaml_string(tag)}" for tag in summary.tags]]
    if not summary.tags:
        tag_lines = ["tags: []"]
    key_point_lines = [
        "key_points:",
        *[f"  - {_yaml_string(point.strip())}" for point in summary.key_points],
    ]
    if not summary.key_points:
        key_point_lines = ["key_points: []"]
    key_moment_lines = [
        "key_moments:",
        *[
            f"  - {_yaml_string(f'{_format_timestamp(moment.timestamp_seconds)} {moment.point}')}"
            for moment in summary.key_moments
        ],
    ]
    if not summary.key_moments:
        key_moment_lines = ["key_moments: []"]
    lines = [
        "---",
        "type: media-summary",
        f"media_id: {_yaml_string(media_id)}",
        f"title: {_yaml_string(result.title)}",
        f"source: {_yaml_string(result.source)}",
        f"source_url: {_yaml_string(result.url)}",
        f"source_item_id: {_yaml_string(result.source_item_id or media_id)}",
        f"creator: {_yaml_string(result.channel_or_show)}",
        f"date_added: {added_at.astimezone(UTC).date().isoformat()}",
        f"published: {published or 'null'}",
        f"duration_seconds: {result.duration_seconds}",
        *tag_lines,
        *key_point_lines,
        *key_moment_lines,
        f"worth_rewatching: {'true' if summary.worth_rewatching else 'false'}",
        f"summary_model: {_yaml_string(summary_model)}",
        f"transcription_model: {_yaml_string(result.transcription_model or 'unknown')}",
        f"transcript_retained: {'true' if transcript_path else 'false'}",
        f"anthropic_requests: {usage.anthropic_requests}",
        f"anthropic_input_tokens: {usage.anthropic_input_tokens}",
        f"anthropic_output_tokens: {usage.anthropic_output_tokens}",
        f"openai_requests: {usage.openai_requests}",
        f"openai_audio_seconds: {round(usage.openai_audio_seconds, 3)}",
        f"estimated_cost_usd: {round(usage.estimated_cost_usd, 6)}",
    ]
    if result.thumbnail_url:
        lines.append(f"thumbnail_url: {_yaml_string(result.thumbnail_url)}")
    lines += [
        "---",
        "",
        f"# {result.title}",
        "",
        f"[{result.source.title()} source]({result.url})",
        "",
        "## TL;DR",
        "",
        summary.tldr.strip(),
        "",
        "## Key Points",
        "",
        *[f"- {point.strip()}" for point in summary.key_points],
    ]
    if summary.key_moments:
        lines += ["", "## Key Moments", ""]
        for moment in summary.key_moments:
            timestamp = _format_timestamp(moment.timestamp_seconds)
            link = _moment_link(result, moment.timestamp_seconds)
            label = f"[{timestamp}]({link})" if link else f"**{timestamp}**"
            lines.append(f"- {label} — {moment.point.strip()}")
    if transcript_path:
        vault_relative = TRANSCRIPT_DIR / transcript_path.name
        lines += ["", "## Transcript", "", f"![[{vault_relative.as_posix()}|Full transcript]]"]
    lines.append("")
    return "\n".join(lines)


def _save_sync(
    result: TranscriptResult,
    summary: Summary,
    vault_path: Path,
    retain_transcript: bool,
    summary_model: str,
    added_at: datetime,
    usage: UsageStats,
) -> str:
    vault_path = vault_path.expanduser().resolve()
    if not vault_path.is_dir() or not (vault_path / ".obsidian").is_dir():
        raise ObsidianError(f"Not an Obsidian vault: {vault_path}")

    media_id = source_id(result)
    transcript_path: Path | None = None
    if retain_transcript:
        transcript_dir = vault_path / TRANSCRIPT_DIR
        transcript_path = _find_existing(transcript_dir, f"{media_id}-transcript")
        if transcript_path is None:
            transcript_path = transcript_dir / f"{media_id}-transcript.md"
            _atomic_write_once(transcript_path, _render_transcript(result, media_id))

    summary_dir = vault_path / SUMMARY_DIR
    summary_path = _find_existing(summary_dir, media_id)
    if summary_path is None:
        summary_path = summary_dir / f"{media_id}.md"
        content = _render_summary(
            result,
            summary,
            media_id,
            added_at,
            summary_model,
            transcript_path,
            usage,
        )
        _atomic_write_once(summary_path, content)
    return summary_path.relative_to(vault_path).as_posix()


async def save_to_obsidian(
    result: TranscriptResult,
    summary: Summary,
    vault_path: str,
    *,
    retain_transcript: bool,
    summary_model: str,
    added_at: datetime | None = None,
    usage: UsageStats | None = None,
) -> str:
    """Save a summary and optional transcript, returning the vault-relative note path."""
    timestamp = added_at or datetime.now(UTC)
    return await asyncio.to_thread(
        _save_sync,
        result,
        summary,
        Path(vault_path),
        retain_transcript,
        summary_model,
        timestamp,
        usage or UsageStats(),
    )

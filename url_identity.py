"""Canonical submission identities used to suppress duplicate jobs."""

import hashlib
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

_TRACKING_PARAMS = {"si"}


def submission_identity(url: str) -> tuple[str, bool]:
    """Return (dedupe_key, reuse_completed) for a submitted URL."""
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower().removeprefix("www.")
    query = parse_qs(parsed.query)

    if hostname in {"youtube.com", "m.youtube.com", "youtu.be"}:
        if hostname == "youtu.be":
            video_id = parsed.path.strip("/").split("/")[0]
        else:
            video_id = query.get("v", [""])[0]
        if video_id:
            return f"youtube:{video_id}", True

    if hostname == "podcasts.apple.com":
        podcast_id = next(
            (part[2:] for part in parsed.path.split("/") if part.startswith("id")),
            "",
        )
        episode_id = query.get("i", [""])[0]
        if podcast_id and episode_id:
            return f"apple-podcast:{podcast_id}:episode:{episode_id}", True

    if hostname in {"vimeo.com", "player.vimeo.com"}:
        video_id = next(
            (part for part in reversed(parsed.path.split("/")) if part.isdigit()),
            "",
        )
        if video_id:
            return f"vimeo:{video_id}", True

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMS
    ]
    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            "",
            urlencode(sorted(filtered_query)),
            "",
        )
    )
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    is_direct_media = parsed.path.lower().endswith(
        (".m4a", ".mov", ".mp3", ".mp4", ".wav", ".webm")
    )
    return f"url:{digest}", is_direct_media

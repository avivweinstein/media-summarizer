"""Canonical submission identities used to suppress duplicate jobs."""

import hashlib
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

_TRACKING_PARAMS = {"si"}
_TWITTER_HOSTNAMES = {
    "m.twitter.com",
    "m.x.com",
    "mobile.twitter.com",
    "mobile.x.com",
    "twitter.com",
    "x.com",
}


def twitter_status_parts(url: str) -> tuple[str, str | None] | None:
    """Return the post ID and optional selected media index for an X/Twitter URL."""
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower().removeprefix("www.")
    if hostname not in _TWITTER_HOSTNAMES:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    try:
        status_position = next(
            index for index, part in enumerate(parts) if part in {"status", "statuses"}
        )
        status_id = parts[status_position + 1]
    except (StopIteration, IndexError):
        return None
    if not status_id.isdigit():
        return None
    media_index = None
    if len(parts) > status_position + 3 and parts[status_position + 2] in {"photo", "video"}:
        candidate = parts[status_position + 3]
        media_index = candidate if candidate.isdigit() else None
    return status_id, media_index


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

    if twitter_parts := twitter_status_parts(url):
        status_id, media_index = twitter_parts
        suffix = f":media:{media_index}" if media_index else ""
        return f"twitter:{status_id}{suffix}", True

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

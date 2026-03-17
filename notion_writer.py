"""Notion API writer.

Creates a Notion page in the configured database with all properties from the
design schema and a clean markdown body (TL;DR + key points as blocks).
"""

import logging
from datetime import UTC, datetime

from notion_client import AsyncClient
from notion_client.errors import APIResponseError

from config import settings
from exceptions import NotionError
from models import Summary, TranscriptResult

logger = logging.getLogger(__name__)

_SOURCE_DISPLAY: dict[str, str] = {
    "youtube": "YouTube",
    "podcast": "Podcast",
    "spotify": "Spotify",
}

# Notion API max chars per single rich_text content object
_RICH_TEXT_LIMIT = 2000


def _rich_text(content: str) -> list[dict[str, object]]:
    """Wrap a string in a Notion rich_text array, truncating if over the API limit."""
    if len(content) > _RICH_TEXT_LIMIT:
        content = content[: _RICH_TEXT_LIMIT - 3] + "..."
    return [{"type": "text", "text": {"content": content}}]


def _build_properties(
    result: TranscriptResult, summary: Summary
) -> dict[str, object]:
    """Build the Notion page properties dict from a TranscriptResult and Summary."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    source_label = _SOURCE_DISPLAY.get(result.source, result.source.title())

    props: dict[str, object] = {
        "Title": {"title": _rich_text(result.title)},
        "URL": {"url": result.url},
        "Source": {"select": {"name": source_label}},
        "Channel / Show": {"rich_text": _rich_text(result.channel_or_show)},
        "Date Added": {"date": {"start": today}},
        "Duration": {"number": result.duration_seconds},
        "Tags": {"multi_select": [{"name": tag} for tag in summary.tags]},
        "Worth Rewatching": {"checkbox": summary.worth_rewatching},
        "TL;DR": {"rich_text": _rich_text(summary.tldr)},
    }

    if result.published_at:
        props["Published"] = {"date": {"start": result.published_at.strftime("%Y-%m-%d")}}

    if result.thumbnail_url:
        props["Thumbnail"] = {"url": result.thumbnail_url}

    return props


def _build_children(summary: Summary) -> list[dict[str, object]]:
    """Build Notion block children for the page body."""
    blocks: list[dict[str, object]] = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": _rich_text("TL;DR")},
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": _rich_text(summary.tldr)},
        },
    ]

    if summary.key_points:
        blocks.append({"object": "block", "type": "divider", "divider": {}})
        blocks.append(
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": _rich_text("Key Points")},
            }
        )
        for point in summary.key_points:
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": _rich_text(point)},
                }
            )

    return blocks


async def save_to_notion(
    result: TranscriptResult, summary: Summary, job_id: str = "-"
) -> str:
    """Create a Notion page and return its page ID.

    Raises NotionError on any API failure (wrong database ID, auth error, etc.).
    """
    log = f"job_id={job_id} url={result.url[:60]!r} source={result.source}"
    logger.info("%s event=notion_save_start title=%r", log, result.title)

    client = AsyncClient(auth=settings.notion_api_key)

    try:
        response = await client.pages.create(
            parent={"database_id": settings.notion_database_id},
            properties=_build_properties(result, summary),
            children=_build_children(summary),
        )
    except APIResponseError as e:
        raise NotionError(
            f"Notion API error {e.code} (HTTP {e.status}): {e}"
        ) from e
    except Exception as e:
        raise NotionError(f"Failed to save to Notion: {e}") from e

    page_id: str = response["id"]
    logger.info("%s event=notion_save_done page_id=%s", log, page_id)
    return page_id

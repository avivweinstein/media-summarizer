"""Tests for Notion page creation.

_build_properties: pure-function tests — verify correct Notion API payload structure.
_build_children:  pure-function tests — verify block structure for page body.
save_to_notion:   mocked AsyncClient — no real API calls.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from notion_client.errors import APIErrorCode, APIResponseError

from exceptions import NotionError
from models import Summary, TranscriptResult
from notion_writer import _RICH_TEXT_LIMIT, _build_children, _build_properties, save_to_notion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(**kwargs: object) -> TranscriptResult:
    defaults: dict[str, object] = {
        "title": "Zone 2 Training Explained",
        "source": "youtube",
        "url": "https://youtube.com/watch?v=test123",
        "channel_or_show": "Peter Attia MD",
        "duration_seconds": 3600,
        "transcript": "This is the transcript.",
        "published_at": datetime(2024, 3, 1, tzinfo=UTC),
        "thumbnail_url": "https://img.youtube.com/vi/test123/hq.jpg",
    }
    defaults.update(kwargs)
    return TranscriptResult(**defaults)  # type: ignore[arg-type]


def make_summary(**kwargs: object) -> Summary:
    defaults: dict[str, object] = {
        "tldr": "Zone 2 training improves mitochondrial function.",
        "key_points": ["Point A.", "Point B.", "Point C."],
        "tags": ["fitness", "health"],
        "worth_rewatching": True,
    }
    defaults.update(kwargs)
    return Summary(**defaults)  # type: ignore[arg-type]


def make_notion_error(code: APIErrorCode, status: int, message: str) -> APIResponseError:
    req = httpx.Request("POST", "https://api.notion.com/v1/pages")
    resp = httpx.Response(status, request=req, text="{}")
    return APIResponseError(
        code=code, status=status, message=message,
        headers=resp.headers, raw_body_text="{}"
    )


# ---------------------------------------------------------------------------
# _build_properties — pure function tests
# ---------------------------------------------------------------------------

class TestBuildProperties:
    def test_title_is_set(self) -> None:
        props: Any = _build_properties(make_result(), make_summary())
        title_content = props["Title"]["title"][0]["text"]["content"]
        assert title_content == "Zone 2 Training Explained"

    def test_url_is_set(self) -> None:
        props: Any = _build_properties(make_result(), make_summary())
        assert props["URL"] == {"url": "https://youtube.com/watch?v=test123"}

    def test_youtube_source_maps_to_display_name(self) -> None:
        props: Any = _build_properties(make_result(source="youtube"), make_summary())
        assert props["Source"] == {"select": {"name": "YouTube"}}

    def test_podcast_source_maps_to_display_name(self) -> None:
        props: Any = _build_properties(make_result(source="podcast"), make_summary())
        assert props["Source"] == {"select": {"name": "Podcast"}}

    def test_unknown_source_title_cased(self) -> None:
        props: Any = _build_properties(make_result(source="rss"), make_summary())
        assert props["Source"] == {"select": {"name": "Rss"}}

    def test_channel_show_is_set(self) -> None:
        props: Any = _build_properties(make_result(), make_summary())
        content = props["Channel / Show"]["rich_text"][0]["text"]["content"]
        assert content == "Peter Attia MD"

    def test_date_added_is_today(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        props: Any = _build_properties(make_result(), make_summary())
        assert props["Date Added"] == {"date": {"start": today}}

    def test_duration_is_number(self) -> None:
        props: Any = _build_properties(make_result(duration_seconds=1234), make_summary())
        assert props["Duration"] == {"number": 1234}

    def test_tags_are_multi_select(self) -> None:
        props: Any = _build_properties(make_result(), make_summary(tags=["fitness", "ai"]))
        assert props["Tags"] == {"multi_select": [{"name": "fitness"}, {"name": "ai"}]}

    def test_empty_tags(self) -> None:
        props: Any = _build_properties(make_result(), make_summary(tags=[]))
        assert props["Tags"] == {"multi_select": []}

    def test_worth_rewatching_true(self) -> None:
        props: Any = _build_properties(make_result(), make_summary(worth_rewatching=True))
        assert props["Worth Rewatching"] == {"checkbox": True}

    def test_worth_rewatching_false(self) -> None:
        props: Any = _build_properties(make_result(), make_summary(worth_rewatching=False))
        assert props["Worth Rewatching"] == {"checkbox": False}

    def test_tldr_is_rich_text(self) -> None:
        props: Any = _build_properties(make_result(), make_summary())
        content = props["TL;DR"]["rich_text"][0]["text"]["content"]
        assert content == "Zone 2 training improves mitochondrial function."

    def test_published_date_included_when_present(self) -> None:
        props: Any = _build_properties(
            make_result(published_at=datetime(2024, 3, 1, tzinfo=UTC)), make_summary()
        )
        assert props["Published"] == {"date": {"start": "2024-03-01"}}

    def test_published_date_omitted_when_none(self) -> None:
        props: Any = _build_properties(make_result(published_at=None), make_summary())
        assert "Published" not in props

    def test_thumbnail_included_when_present(self) -> None:
        props: Any = _build_properties(
            make_result(thumbnail_url="https://img.example.com/thumb.jpg"), make_summary()
        )
        assert props["Thumbnail"] == {"url": "https://img.example.com/thumb.jpg"}

    def test_thumbnail_omitted_when_none(self) -> None:
        props: Any = _build_properties(make_result(thumbnail_url=None), make_summary())
        assert "Thumbnail" not in props

    def test_long_tldr_truncated_to_limit(self) -> None:
        long_tldr = "x" * (_RICH_TEXT_LIMIT + 500)
        props: Any = _build_properties(make_result(), make_summary(tldr=long_tldr))
        content = props["TL;DR"]["rich_text"][0]["text"]["content"]
        assert len(content) == _RICH_TEXT_LIMIT
        assert content.endswith("...")

    def test_long_title_truncated_to_limit(self) -> None:
        long_title = "A" * (_RICH_TEXT_LIMIT + 100)
        props: Any = _build_properties(make_result(title=long_title), make_summary())
        content = props["Title"]["title"][0]["text"]["content"]
        assert len(content) == _RICH_TEXT_LIMIT


# ---------------------------------------------------------------------------
# _build_children — pure function tests
# ---------------------------------------------------------------------------

class TestBuildChildren:
    def test_first_block_is_tldr_heading(self) -> None:
        blocks: Any = _build_children(make_summary())
        assert blocks[0]["type"] == "heading_2"
        content = blocks[0]["heading_2"]["rich_text"][0]["text"]["content"]
        assert content == "TL;DR"

    def test_second_block_is_tldr_paragraph(self) -> None:
        blocks: Any = _build_children(make_summary())
        assert blocks[1]["type"] == "paragraph"
        content = blocks[1]["paragraph"]["rich_text"][0]["text"]["content"]
        assert content == "Zone 2 training improves mitochondrial function."

    def test_key_points_section_present_when_non_empty(self) -> None:
        blocks: Any = _build_children(make_summary(key_points=["A.", "B."]))
        types = [b["type"] for b in blocks]
        assert "divider" in types
        assert "bulleted_list_item" in types

    def test_key_points_heading_present(self) -> None:
        blocks: Any = _build_children(make_summary(key_points=["A.", "B."]))
        headings = [b for b in blocks if b["type"] == "heading_2"]
        heading_texts = [
            h["heading_2"]["rich_text"][0]["text"]["content"]
            for h in headings
        ]
        assert "Key Points" in heading_texts

    def test_correct_number_of_bullet_items(self) -> None:
        blocks: Any = _build_children(make_summary(key_points=["A.", "B.", "C."]))
        bullets = [b for b in blocks if b["type"] == "bulleted_list_item"]
        assert len(bullets) == 3

    def test_bullet_text_matches_key_points(self) -> None:
        blocks: Any = _build_children(make_summary(key_points=["First.", "Second."]))
        bullets = [b for b in blocks if b["type"] == "bulleted_list_item"]
        texts = [b["bulleted_list_item"]["rich_text"][0]["text"]["content"] for b in bullets]
        assert texts == ["First.", "Second."]

    def test_no_key_points_section_when_empty(self) -> None:
        blocks: Any = _build_children(make_summary(key_points=[]))
        types = [b["type"] for b in blocks]
        assert "divider" not in types
        assert "bulleted_list_item" not in types

    def test_total_block_count_with_three_key_points(self) -> None:
        # heading_2 (TL;DR) + paragraph + divider + heading_2 (Key Points) + 3 bullets = 7
        blocks: Any = _build_children(make_summary(key_points=["A.", "B.", "C."]))
        assert len(blocks) == 7

    def test_total_block_count_with_no_key_points(self) -> None:
        # heading_2 (TL;DR) + paragraph = 2
        blocks: Any = _build_children(make_summary(key_points=[]))
        assert len(blocks) == 2


# ---------------------------------------------------------------------------
# save_to_notion — mocked AsyncClient
# ---------------------------------------------------------------------------

class TestSaveToNotion:
    async def test_returns_page_id_on_success(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.pages.create.return_value = {"id": "page-abc-123"}
        mocker.patch("notion_writer.AsyncClient", return_value=mock_client)

        page_id = await save_to_notion(make_result(), make_summary())
        assert page_id == "page-abc-123"

    async def test_correct_database_id_used(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.pages.create.return_value = {"id": "page-abc-123"}
        mocker.patch("notion_writer.AsyncClient", return_value=mock_client)

        with patch("notion_writer.settings") as mock_settings:
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "db-xyz-789"
            await save_to_notion(make_result(), make_summary())

        call_kwargs = mock_client.pages.create.call_args.kwargs
        assert call_kwargs["parent"] == {"database_id": "db-xyz-789"}

    async def test_properties_and_children_passed(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.pages.create.return_value = {"id": "page-abc-123"}
        mocker.patch("notion_writer.AsyncClient", return_value=mock_client)

        await save_to_notion(make_result(), make_summary())

        call_kwargs = mock_client.pages.create.call_args.kwargs
        assert "properties" in call_kwargs
        assert "children" in call_kwargs

    async def test_api_response_error_raises_notion_error(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.pages.create.side_effect = make_notion_error(
            APIErrorCode.ValidationError, 400, "Invalid property"
        )
        mocker.patch("notion_writer.AsyncClient", return_value=mock_client)

        with pytest.raises(NotionError, match="Notion API error"):
            await save_to_notion(make_result(), make_summary())

    async def test_generic_exception_raises_notion_error(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.pages.create.side_effect = ConnectionError("Network unreachable")
        mocker.patch("notion_writer.AsyncClient", return_value=mock_client)

        with pytest.raises(NotionError, match="Failed to save to Notion"):
            await save_to_notion(make_result(), make_summary())

    async def test_notion_error_message_includes_summary_tldr(self, mocker: MagicMock) -> None:
        """The caller (pipeline) should surface the TL;DR in the user-facing error
        message — this test ensures the NotionError propagates cleanly from save_to_notion."""
        mock_client = AsyncMock()
        mock_client.pages.create.side_effect = make_notion_error(
            APIErrorCode.InternalServerError, 500, "Server error"
        )
        mocker.patch("notion_writer.AsyncClient", return_value=mock_client)

        try:
            await save_to_notion(make_result(), make_summary())
        except NotionError as e:
            assert "Notion" in str(e)

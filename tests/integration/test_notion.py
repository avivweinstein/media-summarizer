"""Notion integration tests — create and verify pages in the test database."""

from datetime import UTC, datetime

import pytest

from exceptions import NotionError
from models import Summary, TranscriptResult
from notion_writer import save_to_notion

pytestmark = pytest.mark.integration


def _sample_result() -> TranscriptResult:
    return TranscriptResult(
        title="Integration Test Episode",
        source="youtube",
        url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        channel_or_show="Test Channel",
        duration_seconds=19,
        transcript="Just some elephants at the zoo.",
        published_at=datetime(2005, 4, 23, tzinfo=UTC),
    )


def _sample_summary() -> Summary:
    return Summary(
        tldr="A short video of elephants at the zoo.",
        key_points=["Elephants are shown", "It is a zoo"],
        tags=["tech"],
        worth_rewatching=False,
    )


async def test_save_returns_page_id(notion_test_db_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from config import settings
    monkeypatch.setattr(settings, "notion_database_id", notion_test_db_id)

    page_id = await save_to_notion(_sample_result(), _sample_summary(), job_id="integration-test")

    assert page_id
    assert len(page_id) > 0


async def test_saved_page_id_looks_like_uuid(
    notion_test_db_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from config import settings
    monkeypatch.setattr(settings, "notion_database_id", notion_test_db_id)

    page_id = await save_to_notion(_sample_result(), _sample_summary(), job_id="integration-test")

    # Notion page IDs are UUIDs (with or without dashes)
    clean = page_id.replace("-", "")
    assert len(clean) == 32
    assert clean.isalnum()


async def test_invalid_database_id_raises_notion_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import settings
    monkeypatch.setattr(settings, "notion_database_id", "00000000000000000000000000000000")

    with pytest.raises(NotionError):
        await save_to_notion(_sample_result(), _sample_summary(), job_id="integration-test")

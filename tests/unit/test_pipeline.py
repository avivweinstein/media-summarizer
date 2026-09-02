"""Tests for pipeline retry logic and webhook notifications."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import job_queue
from config import settings
from models import JobStatus, Summary, TranscriptResult
from pipeline import run_job


@pytest.fixture(autouse=True)
def _configure_output_defaults(mocker: MagicMock) -> None:
    mocker.patch.object(settings, "notion_enabled", True)
    mocker.patch.object(settings, "obsidian_vault_path", "")


def _transcript() -> TranscriptResult:
    return TranscriptResult(
        title="Test Episode",
        source="youtube",
        url="https://youtube.com/watch?v=test",
        channel_or_show="Test Channel",
        duration_seconds=300,
        transcript="This is a test transcript.",
    )


def _summary() -> Summary:
    return Summary(
        tldr="A great test episode.",
        key_points=["Point 1", "Point 2"],
        tags=["tech"],
        worth_rewatching=True,
    )


def _mock_happy_path(mocker: MagicMock, source_type: str = "youtube") -> None:
    """Patch all external calls to succeed."""
    mocker.patch("pipeline.detect_source", return_value=source_type)
    mock_source = AsyncMock()
    mock_source.fetch.return_value = _transcript()
    if source_type == "youtube":
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)
    else:
        mocker.patch("pipeline.PodcastSource", return_value=mock_source)
    mocker.patch("pipeline.summarize", return_value=_summary())
    mocker.patch("pipeline.save_to_notion", return_value="page-abc-123")
    mocker.patch(
        "pipeline.save_to_obsidian",
        return_value="Generated/Summaries/youtube-test.md",
    )


def _mock_webhook(mocker: MagicMock) -> AsyncMock:
    """Patch httpx so webhook POSTs are captured. Returns the mock client.

    The client is set as its own __aenter__ return value so that
    `async with httpx.AsyncClient(...) as client:` yields this same mock,
    and assertions on mock_client.post are correct.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post.return_value = mock_resp
    mocker.patch("pipeline.httpx.AsyncClient", return_value=mock_client)
    return mock_client


class TestRetryLogic:
    async def test_succeeds_on_first_attempt(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123", db_path=db_path
        )
        _mock_happy_path(mocker)
        mocker.patch("pipeline._notify_webhook")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done
        assert result.retry_count == 0

    async def test_retries_on_transient_failure_and_succeeds(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123", db_path=db_path
        )
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mock_source = AsyncMock()
        mock_source.fetch.side_effect = [RuntimeError("transient"), _transcript()]
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)
        mocker.patch("pipeline.summarize", return_value=_summary())
        mocker.patch("pipeline.save_to_notion", return_value="page-123")
        mocker.patch("pipeline._notify_webhook")
        mocker.patch("pipeline.asyncio.sleep")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done
        assert result.retry_count == 1  # second attempt (0-indexed)

    async def test_exhausted_retries_marks_failed(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123", db_path=db_path
        )
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mock_source = AsyncMock()
        mock_source.fetch.side_effect = RuntimeError("always fails")
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)
        mocker.patch("pipeline._notify_webhook")
        mocker.patch("pipeline.asyncio.sleep")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.failed
        assert "always fails" in (result.error or "")

    async def test_error_message_is_from_last_attempt(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123", db_path=db_path
        )
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mock_source = AsyncMock()
        mock_source.fetch.side_effect = [
            RuntimeError("first error"),
            RuntimeError("second error"),
            RuntimeError("final error"),
        ]
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)
        mocker.patch("pipeline._notify_webhook")
        mocker.patch("pipeline.asyncio.sleep")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.error == "final error"

    async def test_retry_uses_increasing_backoff(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123", db_path=db_path
        )
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mock_source = AsyncMock()
        mock_source.fetch.side_effect = RuntimeError("fail")
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)
        mocker.patch("pipeline._notify_webhook")
        sleep_mock = mocker.patch("pipeline.asyncio.sleep")

        await run_job(job.job_id, db_path=db_path)

        # Two sleeps: before attempt 2 and before attempt 3
        assert sleep_mock.call_count == 2
        times = [call.args[0] for call in sleep_mock.call_args_list]
        assert times[0] < times[1]  # backoff increases

    async def test_missing_job_returns_without_error(
        self, db_path: str
    ) -> None:
        await run_job("nonexistent-id", db_path=db_path)


class TestOutputRouting:
    async def test_obsidian_can_be_the_only_output(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123", db_path=db_path
        )
        _mock_happy_path(mocker)
        mocker.patch("pipeline._notify_webhook")
        mock_settings = mocker.patch("pipeline.settings")
        mock_settings.youtube_api_key = ""
        mock_settings.obsidian_vault_path = "/vault"
        mock_settings.obsidian_retain_transcript = True
        mock_settings.notion_enabled = False
        notion = mocker.patch("pipeline.save_to_notion")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done
        assert result.obsidian_note_path == "Generated/Summaries/youtube-test.md"
        notion.assert_not_called()

    async def test_notion_failure_is_non_blocking_after_obsidian_success(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123", db_path=db_path
        )
        _mock_happy_path(mocker)
        mocker.patch("pipeline._notify_webhook")
        mock_settings = mocker.patch("pipeline.settings")
        mock_settings.youtube_api_key = ""
        mock_settings.obsidian_vault_path = "/vault"
        mock_settings.obsidian_retain_transcript = True
        mock_settings.notion_enabled = True
        mocker.patch("pipeline.save_to_notion", side_effect=RuntimeError("Notion unavailable"))

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done
        assert result.obsidian_note_path is not None
        assert result.notion_page_id is None
        assert result.notion_error == "Notion unavailable"

    async def test_job_fails_when_no_output_is_configured(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123", db_path=db_path
        )
        _mock_happy_path(mocker)
        mocker.patch("pipeline._notify_webhook")
        mocker.patch("pipeline.asyncio.sleep")
        mock_settings = mocker.patch("pipeline.settings")
        mock_settings.youtube_api_key = ""
        mock_settings.obsidian_vault_path = ""
        mock_settings.notion_enabled = False

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.failed
        assert result.error == "No configured output destination succeeded."

    async def test_notion_only_failure_preserves_original_error(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123", db_path=db_path
        )
        _mock_happy_path(mocker)
        mocker.patch("pipeline._notify_webhook")
        mocker.patch("pipeline.asyncio.sleep")
        mock_settings = mocker.patch("pipeline.settings")
        mock_settings.youtube_api_key = ""
        mock_settings.obsidian_vault_path = ""
        mock_settings.notion_enabled = True
        mocker.patch("pipeline.save_to_notion", side_effect=RuntimeError("Notion unavailable"))

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.failed
        assert result.error == "Notion unavailable"


class TestWebhookNotifications:
    async def test_webhook_fired_on_success(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            webhook_url="https://hooks.example.com/cb",
            db_path=db_path,
        )
        _mock_happy_path(mocker)
        mock_client = _mock_webhook(mocker)

        await run_job(job.job_id, db_path=db_path)

        mock_client.post.assert_called_once()
        posted_url: str = mock_client.post.call_args.args[0]
        payload: dict[str, object] = mock_client.post.call_args.kwargs["json"]
        assert "hooks.example.com" in posted_url
        assert payload["status"] == "done"
        assert payload["tldr"] == "A great test episode."
        assert "notion_url" in payload

    async def test_webhook_payload_contains_notion_url(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            webhook_url="https://hooks.example.com/cb",
            db_path=db_path,
        )
        _mock_happy_path(mocker)
        mock_client = _mock_webhook(mocker)

        await run_job(job.job_id, db_path=db_path)

        payload: dict[str, object] = mock_client.post.call_args.kwargs["json"]
        assert payload["notion_url"] == "https://www.notion.so/pageabc123"

    async def test_webhook_fired_on_final_failure(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            webhook_url="https://hooks.example.com/cb",
            db_path=db_path,
        )
        mocker.patch("pipeline.detect_source", side_effect=RuntimeError("boom"))
        mocker.patch("pipeline.asyncio.sleep")  # skip real backoff delays
        mock_client = _mock_webhook(mocker)

        await run_job(job.job_id, db_path=db_path)

        mock_client.post.assert_called_once()
        payload: dict[str, object] = mock_client.post.call_args.kwargs["json"]
        assert payload["status"] == "failed"
        assert "boom" in str(payload["error"])

    async def test_no_webhook_when_no_url_configured(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            webhook_url=None,
            db_path=db_path,
        )
        _mock_happy_path(mocker)
        http_patch = mocker.patch("pipeline.httpx.AsyncClient")
        mock_settings = mocker.patch("pipeline.settings")
        mock_settings.openclaw_webhook_url = ""
        mock_settings.youtube_api_key = "fake-key"

        await run_job(job.job_id, db_path=db_path)

        http_patch.assert_not_called()

    async def test_webhook_failure_does_not_fail_job(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            webhook_url="https://hooks.example.com/cb",
            db_path=db_path,
        )
        _mock_happy_path(mocker)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post.side_effect = httpx.ConnectError("no route")
        mocker.patch("pipeline.httpx.AsyncClient", return_value=mock_client)

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done  # job done despite webhook failure

    async def test_webhook_fired_only_once_on_eventual_success(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        """Webhook fires once when job succeeds on second attempt."""
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            webhook_url="https://hooks.example.com/cb",
            db_path=db_path,
        )
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mock_source = AsyncMock()
        mock_source.fetch.side_effect = [RuntimeError("first"), _transcript()]
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)
        mocker.patch("pipeline.summarize", return_value=_summary())
        mocker.patch("pipeline.save_to_notion", return_value="page-abc-123")
        mocker.patch("pipeline.asyncio.sleep")
        mock_client = _mock_webhook(mocker)

        await run_job(job.job_id, db_path=db_path)

        assert mock_client.post.call_count == 1
        payload: dict[str, object] = mock_client.post.call_args.kwargs["json"]
        assert payload["status"] == "done"

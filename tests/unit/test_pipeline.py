"""Tests for pipeline retry logic and webhook notifications."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import job_queue
from config import settings
from exceptions import UsageLimitError
from models import JobStage, JobStatus, Summary, TranscriptResult, UsageStats
from pipeline import _notify_webhook, run_job


@pytest.fixture(autouse=True)
def _configure_output_defaults(mocker: MagicMock) -> None:
    mocker.patch.object(settings, "notion_enabled", True)
    mocker.patch.object(settings, "obsidian_vault_path", "")
    mocker.patch.object(settings, "webhooks_enabled", True)


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
    source_classes = {
        "youtube": "YouTubeSource",
        "podcast": "PodcastSource",
        "media": "MediaSource",
        "article": "ArticleSource",
        "upload": "UploadSource",
    }
    mocker.patch(f"pipeline.{source_classes[source_type]}", return_value=mock_source)
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
    async def test_restart_reuses_persisted_transcript_and_summary(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            processing_mode="local",
            external_processing_approved=False,
            db_path=db_path,
        )
        job.status = JobStatus.processing
        job.stage = JobStage.saving_obsidian
        job.result = _transcript()
        job.summary = _summary()
        await job_queue.update_job(job, db_path=db_path)
        await job_queue.recover_incomplete_jobs(db_path=db_path)

        source = mocker.patch("pipeline.YouTubeSource")
        summarize_mock = mocker.patch("pipeline.summarize")
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mocker.patch("pipeline._notify_webhook")
        save = mocker.patch(
            "pipeline.save_to_obsidian",
            return_value="Generated/Summaries/youtube-test.md",
        )
        mocker.patch.object(settings, "obsidian_vault_path", "/vault")
        mocker.patch.object(settings, "notion_enabled", False)

        await run_job(job.job_id, db_path=db_path)

        source.assert_not_called()
        summarize_mock.assert_not_called()
        save.assert_awaited_once()
        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done
        assert result.retry_count == 1

    async def test_interrupted_notion_save_is_not_replayed(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
        job.status = JobStatus.processing
        job.stage = JobStage.saving_notion
        job.result = _transcript()
        job.summary = _summary()
        job.obsidian_note_path = "Generated/Summaries/youtube-test.md"
        await job_queue.update_job(job, db_path=db_path)
        await job_queue.recover_incomplete_jobs(db_path=db_path)

        mocker.patch("pipeline.detect_source", return_value="youtube")
        notion = mocker.patch("pipeline.save_to_notion")
        mocker.patch("pipeline._notify_webhook")
        mocker.patch.object(settings, "obsidian_vault_path", "/vault")
        mocker.patch.object(settings, "notion_enabled", True)

        await run_job(job.job_id, db_path=db_path)

        notion.assert_not_called()
        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done
        assert result.notion_error is not None

    async def test_provider_usage_reservation_survives_worker_cancellation(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mock_source = AsyncMock()

        async def reserve_then_cancel(*_args: object, **kwargs: object) -> TranscriptResult:
            usage = cast(UsageStats, kwargs["usage"])
            usage.openai_requests += 1
            persist = cast(
                Callable[[UsageStats], Awaitable[None]],
                kwargs["persist_usage"],
            )
            await persist(usage)
            raise asyncio.CancelledError

        mock_source.fetch.side_effect = reserve_then_cancel
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)

        with pytest.raises(asyncio.CancelledError):
            await run_job(job.job_id, db_path=db_path)

        recovered = await job_queue.get_job(job.job_id, db_path=db_path)
        assert recovered is not None
        assert recovered.usage.openai_requests == 1

    @pytest.mark.parametrize("cancel_point", ["fetch", "summarize"])
    async def test_database_cancellation_during_work_is_not_resurrected(
        self,
        cancel_point: str,
        db_path: str,
        mocker: MagicMock,
    ) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
        mocker.patch("pipeline.detect_source", return_value="youtube")
        source = AsyncMock()

        async def cancel_job() -> None:
            current = await job_queue.get_job(job.job_id, db_path=db_path)
            assert current is not None
            current.status = JobStatus.cancelled
            await job_queue.update_job(current, db_path=db_path)

        async def fetch(*_args: object, **_kwargs: object) -> TranscriptResult:
            if cancel_point == "fetch":
                await cancel_job()
            return _transcript()

        async def summarize_result(*_args: object, **_kwargs: object) -> Summary:
            if cancel_point == "summarize":
                await cancel_job()
            return _summary()

        source.fetch.side_effect = fetch
        mocker.patch("pipeline.YouTubeSource", return_value=source)
        summary_call = mocker.patch("pipeline.summarize", side_effect=summarize_result)
        notion = mocker.patch("pipeline.save_to_notion")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.cancelled
        notion.assert_not_called()
        if cancel_point == "fetch":
            summary_call.assert_not_called()

    @pytest.mark.parametrize("cancel_point", ["obsidian", "notion"])
    async def test_cancellation_during_output_is_not_resurrected_or_notified(
        self,
        cancel_point: str,
        db_path: str,
        mocker: MagicMock,
    ) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
        _mock_happy_path(mocker)
        webhook = mocker.patch("pipeline._notify_webhook")

        async def cancel_job() -> None:
            current = await job_queue.get_job(job.job_id, db_path=db_path)
            assert current is not None
            current.status = JobStatus.cancelled
            await job_queue.update_job(current, db_path=db_path)

        async def save_obsidian(*_args: object, **_kwargs: object) -> str:
            await cancel_job()
            return "Generated/Summaries/youtube-test.md"

        async def save_notion(*_args: object, **_kwargs: object) -> str:
            await cancel_job()
            return "created-before-cancellation-was-observed"

        if cancel_point == "obsidian":
            mocker.patch.object(settings, "obsidian_vault_path", "/vault")
            mocker.patch("pipeline.save_to_obsidian", side_effect=save_obsidian)
            notion = mocker.patch("pipeline.save_to_notion")
        else:
            mocker.patch.object(settings, "obsidian_vault_path", "/vault")
            notion = mocker.patch("pipeline.save_to_notion", side_effect=save_notion)

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.cancelled
        webhook.assert_not_called()
        if cancel_point == "obsidian":
            assert result.obsidian_note_path == "Generated/Summaries/youtube-test.md"
            notion.assert_not_called()
        else:
            assert result.notion_page_id == "created-before-cancellation-was-observed"

            retry = await job_queue.create_job(
                "https://youtube.com/watch?v=abc123", db_path=db_path
            )
            await run_job(retry.job_id, db_path=db_path)
            retried = await job_queue.get_job(retry.job_id, db_path=db_path)
            assert retried is not None
            assert retried.status == JobStatus.done
            assert retried.notion_page_id == "created-before-cancellation-was-observed"
            assert notion.await_count == 1

    async def test_usage_limit_failure_is_not_retried(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
        _mock_happy_path(mocker)
        summarize_mock = mocker.patch(
            "pipeline.summarize",
            side_effect=UsageLimitError("Configured limit reached"),
        )
        mocker.patch("pipeline._notify_webhook")
        sleep_mock = mocker.patch("pipeline.asyncio.sleep")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.failed
        assert result.retry_count == 0
        summarize_mock.assert_awaited_once()
        sleep_mock.assert_not_called()

    async def test_succeeds_on_first_attempt(self, db_path: str, mocker: MagicMock) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
        _mock_happy_path(mocker)
        mocker.patch("pipeline._notify_webhook")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done
        assert result.retry_count == 0

    @pytest.mark.parametrize("source_type", ["article", "media", "upload"])
    async def test_routes_additional_sources(
        self, source_type: str, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job("https://example.com/item", db_path=db_path)
        _mock_happy_path(mocker, source_type)
        mocker.patch("pipeline._notify_webhook")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done

    async def test_local_mode_never_publishes_to_notion(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            processing_mode="local",
            external_processing_approved=False,
            db_path=db_path,
        )
        _mock_happy_path(mocker)
        summarize_mock = mocker.patch("pipeline.summarize", return_value=_summary())
        notion = mocker.patch("pipeline.save_to_notion")
        mocker.patch.object(settings, "processing_mode", "cloud_public")
        mocker.patch.object(settings, "obsidian_vault_path", "/configured-vault")
        mocker.patch("pipeline._notify_webhook")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done
        notion.assert_not_awaited()
        assert summarize_mock.call_args.kwargs["processing_mode"] == "local"

    async def test_unapproved_cloud_job_fails_before_source_fetch(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            processing_mode="cloud_public",
            external_processing_approved=False,
            db_path=db_path,
        )
        source = mocker.patch("pipeline.YouTubeSource")
        summarize_mock = mocker.patch("pipeline.summarize")

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.failed
        assert result.error == "External AI processing was not approved for this job."
        source.assert_not_called()
        summarize_mock.assert_not_called()

    async def test_retries_on_transient_failure_and_succeeds(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
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

    async def test_exhausted_retries_marks_failed(self, db_path: str, mocker: MagicMock) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
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
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
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

    async def test_retry_uses_increasing_backoff(self, db_path: str, mocker: MagicMock) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
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

    async def test_cancellation_during_backoff_stops_retry(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mock_source = AsyncMock()
        mock_source.fetch.side_effect = RuntimeError("retry me")
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)

        async def cancel_during_sleep(_seconds: int) -> None:
            fresh = await job_queue.get_job(job.job_id, db_path=db_path)
            assert fresh is not None
            fresh.status = JobStatus.cancelled
            await job_queue.update_job(fresh, db_path=db_path)

        mocker.patch("pipeline.asyncio.sleep", side_effect=cancel_during_sleep)

        await run_job(job.job_id, db_path=db_path)

        assert mock_source.fetch.await_count == 1
        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.cancelled

    async def test_missing_job_returns_without_error(self, db_path: str) -> None:
        await run_job("nonexistent-id", db_path=db_path)


class TestOutputRouting:
    async def test_usage_is_accumulated_and_persisted(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mock_source = AsyncMock()

        async def fetch_with_usage(*_args: object, **kwargs: object) -> TranscriptResult:
            usage = cast(UsageStats, kwargs["usage"])
            usage.openai_requests += 1
            usage.openai_audio_seconds += 60
            usage.estimated_cost_usd += 0.006
            return _transcript()

        async def summarize_with_usage(*_args: object, **kwargs: object) -> Summary:
            usage = cast(UsageStats, kwargs["usage"])
            usage.anthropic_requests += 1
            usage.anthropic_input_tokens += 100
            usage.estimated_cost_usd += 0.001
            return _summary()

        mock_source.fetch.side_effect = fetch_with_usage
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)
        mocker.patch("pipeline.summarize", side_effect=summarize_with_usage)
        obsidian = mocker.patch(
            "pipeline.save_to_obsidian",
            return_value="Generated/Summaries/youtube-test.md",
        )
        mocker.patch("pipeline._notify_webhook")
        mock_settings = mocker.patch("pipeline.settings")
        mock_settings.youtube_api_key = ""
        mock_settings.obsidian_vault_path = "/vault"
        mock_settings.obsidian_retain_transcript = True
        mock_settings.notion_enabled = False

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.usage.openai_requests == 1
        assert result.usage.anthropic_requests == 1
        assert result.usage.anthropic_input_tokens == 100
        assert result.usage.estimated_cost_usd == pytest.approx(0.007)
        assert obsidian.call_args.kwargs["usage"].estimated_cost_usd == pytest.approx(0.007)

    async def test_obsidian_can_be_the_only_output(self, db_path: str, mocker: MagicMock) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
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
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
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

    async def test_existing_notion_page_for_same_obsidian_note_is_reused(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        prior = await job_queue.create_job("https://example.com/article", db_path=db_path)
        prior.status = JobStatus.done
        prior.obsidian_note_path = "Generated/Summaries/youtube-test.md"
        prior.notion_page_id = "existing-notion-page"
        await job_queue.update_job(prior, db_path=db_path)
        job = await job_queue.create_job("https://example.com/article", db_path=db_path)
        _mock_happy_path(mocker, "article")
        notion = mocker.patch("pipeline.save_to_notion")
        mocker.patch("pipeline._notify_webhook")
        mocker.patch.object(settings, "obsidian_vault_path", "/vault")
        mocker.patch.object(settings, "notion_enabled", True)

        await run_job(job.job_id, db_path=db_path)

        result = await job_queue.get_job(job.job_id, db_path=db_path)
        assert result is not None
        assert result.status == JobStatus.done
        assert result.notion_page_id == "existing-notion-page"
        notion.assert_not_called()

    async def test_job_fails_when_no_output_is_configured(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
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
        job = await job_queue.create_job("https://youtube.com/watch?v=abc123", db_path=db_path)
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
    async def test_local_mode_never_sends_webhook(self, db_path: str, mocker: MagicMock) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            webhook_url="https://hooks.example.com/cb",
            processing_mode="local",
            external_processing_approved=False,
            db_path=db_path,
        )
        http_patch = mocker.patch("pipeline.httpx.AsyncClient")

        await _notify_webhook(job, db_path=db_path)

        http_patch.assert_not_called()

    async def test_webhook_disabled_skips_configured_url(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            webhook_url="https://hooks.example.com/cb",
            db_path=db_path,
        )
        _mock_happy_path(mocker)
        mocker.patch.object(settings, "webhooks_enabled", False)
        http_patch = mocker.patch("pipeline.httpx.AsyncClient")

        await run_job(job.job_id, db_path=db_path)

        http_patch.assert_not_called()

    async def test_webhook_fired_on_success(self, db_path: str, mocker: MagicMock) -> None:
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

    async def test_all_webhook_subscribers_are_notified(
        self, db_path: str, mocker: MagicMock
    ) -> None:
        job = await job_queue.create_job(
            "https://youtube.com/watch?v=abc123",
            webhook_url="https://hooks.example.com/first",
            db_path=db_path,
        )
        mocker.patch("pipeline.detect_source", return_value="youtube")
        mock_source = AsyncMock()

        async def fetch_and_subscribe(*_args: object, **_kwargs: object) -> TranscriptResult:
            await job_queue.create_or_get_job(
                "https://youtu.be/abc123",
                webhook_url="https://hooks.example.com/second",
                db_path=db_path,
            )
            return _transcript()

        mock_source.fetch.side_effect = fetch_and_subscribe
        mocker.patch("pipeline.YouTubeSource", return_value=mock_source)
        mocker.patch("pipeline.summarize", return_value=_summary())
        mocker.patch("pipeline.save_to_notion", return_value="page-abc-123")
        mock_client = _mock_webhook(mocker)

        await run_job(job.job_id, db_path=db_path)

        assert mock_client.post.call_count == 2
        assert [call.args[0] for call in mock_client.post.call_args_list] == [
            "https://hooks.example.com/first",
            "https://hooks.example.com/second",
        ]

    async def test_webhook_fired_on_final_failure(self, db_path: str, mocker: MagicMock) -> None:
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

    async def test_no_webhook_when_no_url_configured(self, db_path: str, mocker: MagicMock) -> None:
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

    async def test_webhook_failure_does_not_fail_job(self, db_path: str, mocker: MagicMock) -> None:
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

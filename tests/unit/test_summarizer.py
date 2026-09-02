"""Tests for the Claude summarizer.

_parse_response: pure-function tests, no mocks needed.
summarize():     mocks the AsyncAnthropic client — no real API calls.
"""

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import APIStatusError
from anthropic.types import TextBlock

from config import settings
from exceptions import SummarizationError, UsageLimitError
from models import Summary, TranscriptResult, TranscriptSegment, UsageStats
from summarizer import MODEL, _chunk_text, _parse_response, summarize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_JSON = json.dumps(
    {
        "tldr": "A great video about cycling training.",
        "key_points": ["Point one.", "Point two.", "Point three."],
        "tags": ["cycling", "fitness"],
        "worth_rewatching": True,
    }
)


def make_result(**kwargs: object) -> TranscriptResult:
    defaults: dict[str, object] = {
        "title": "Test Video",
        "source": "youtube",
        "url": "https://youtube.com/watch?v=test",
        "channel_or_show": "Test Channel",
        "duration_seconds": 300,
        "transcript": "This is a test transcript about cycling and fitness.",
    }
    defaults.update(kwargs)
    return TranscriptResult(**defaults)  # type: ignore[arg-type]


def make_claude_response(content: str) -> MagicMock:
    """Build a mock that looks like an anthropic Messages response.

    Uses a real TextBlock so isinstance checks in summarizer.py pass.
    """
    response = MagicMock()
    response.content = [TextBlock(type="text", text=content)]
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


def make_api_status_error(
    cls: type[APIStatusError], status_code: int, message: str
) -> APIStatusError:
    """Construct an anthropic APIStatusError subclass with a real httpx.Response."""
    mock_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_response = httpx.Response(status_code, request=mock_request)
    return cls(message=message, response=cast(Any, mock_response), body=None)


# ---------------------------------------------------------------------------
# _parse_response — unit tests (no I/O)
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_json_returns_summary(self) -> None:
        s = _parse_response(VALID_JSON)
        assert s.tldr == "A great video about cycling training."
        assert s.key_points == ["Point one.", "Point two.", "Point three."]
        assert s.tags == ["cycling", "fitness"]
        assert s.worth_rewatching is True

    def test_strips_json_markdown_fence(self) -> None:
        fenced = f"```json\n{VALID_JSON}\n```"
        s = _parse_response(fenced)
        assert s.tldr == "A great video about cycling training."

    def test_strips_plain_markdown_fence(self) -> None:
        fenced = f"```\n{VALID_JSON}\n```"
        s = _parse_response(fenced)
        assert s.tldr == "A great video about cycling training."

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(SummarizationError, match="invalid JSON"):
            _parse_response("not json at all {{{")

    def test_json_array_raises(self) -> None:
        with pytest.raises(SummarizationError, match="JSON object"):
            _parse_response("[1, 2, 3]")

    def test_missing_tldr_raises(self) -> None:
        data = {"key_points": ["p1"], "tags": [], "worth_rewatching": False}
        with pytest.raises(SummarizationError, match="tldr"):
            _parse_response(json.dumps(data))

    def test_empty_tldr_raises(self) -> None:
        data = {"tldr": "", "key_points": ["p1"], "tags": [], "worth_rewatching": False}
        with pytest.raises(SummarizationError, match="tldr"):
            _parse_response(json.dumps(data))

    def test_missing_worth_rewatching_raises(self) -> None:
        data = {"tldr": "ok", "key_points": ["p1"], "tags": []}
        with pytest.raises(SummarizationError, match="worth_rewatching"):
            _parse_response(json.dumps(data))

    def test_worth_rewatching_string_true_coerced(self) -> None:
        data = {"tldr": "ok", "key_points": ["p"], "tags": [], "worth_rewatching": "true"}
        assert _parse_response(json.dumps(data)).worth_rewatching is True

    def test_worth_rewatching_string_false_coerced(self) -> None:
        data = {"tldr": "ok", "key_points": ["p"], "tags": [], "worth_rewatching": "false"}
        assert _parse_response(json.dumps(data)).worth_rewatching is False

    def test_worth_rewatching_int_one_coerced(self) -> None:
        data = {"tldr": "ok", "key_points": ["p"], "tags": [], "worth_rewatching": 1}
        assert _parse_response(json.dumps(data)).worth_rewatching is True

    def test_key_points_non_strings_coerced(self) -> None:
        data = {"tldr": "ok", "key_points": [1, 2.5, True], "tags": [], "worth_rewatching": False}
        s = _parse_response(json.dumps(data))
        assert s.key_points == ["1", "2.5", "True"]

    def test_key_points_not_list_raises(self) -> None:
        data = {"tldr": "ok", "key_points": "not a list", "tags": [], "worth_rewatching": False}
        with pytest.raises(SummarizationError, match="key_points"):
            _parse_response(json.dumps(data))

    def test_non_canonical_tags_accepted(self) -> None:
        data = {**json.loads(VALID_JSON), "tags": ["cooking", "travel", "ai"]}
        s = _parse_response(json.dumps(data))
        assert "cooking" in s.tags
        assert "travel" in s.tags

    def test_extra_fields_in_response_ignored(self) -> None:
        data = {**json.loads(VALID_JSON), "surprise_field": "ignored"}
        s = _parse_response(json.dumps(data))
        assert isinstance(s, Summary)

    def test_empty_tags_list_accepted(self) -> None:
        data = {"tldr": "ok", "key_points": ["p"], "tags": [], "worth_rewatching": False}
        s = _parse_response(json.dumps(data))
        assert s.tags == []

    def test_empty_key_points_list_accepted(self) -> None:
        # Claude shouldn't do this but we should not crash
        data = {"tldr": "ok", "key_points": [], "tags": [], "worth_rewatching": False}
        s = _parse_response(json.dumps(data))
        assert s.key_points == []

    def test_valid_key_moments_are_parsed(self) -> None:
        data = {
            **json.loads(VALID_JSON),
            "key_moments": [{"timestamp_seconds": 42, "point": "Core idea."}],
        }

        summary = _parse_response(json.dumps(data))

        assert summary.key_moments[0].timestamp_seconds == 42


class TestChunkText:
    def test_chunks_have_strict_size_limit_and_preserve_text(self) -> None:
        chunks = _chunk_text("one two three four five", 8)

        assert all(len(chunk) <= 8 for chunk in chunks)
        assert " ".join(chunks) == "one two three four five"


# ---------------------------------------------------------------------------
# summarize() — mocked Anthropic client
# ---------------------------------------------------------------------------


class TestSummarize:
    async def test_nvidia_internal_uses_only_internal_endpoint(
        self, mocker: MagicMock
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(VALID_JSON)
        factory = mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        http_client = AsyncMock()
        http_factory = mocker.patch(
            "summarizer.nvidia_http_client",
            return_value=http_client,
        )
        mocker.patch.object(settings, "nvidia_inference_api_key", "nvidia-key")
        mocker.patch.object(
            settings,
            "nvidia_inference_base_url",
            "https://inference-api.nvidia.com",
        )
        mocker.patch.object(settings, "nvidia_inference_model", "internal-model")
        mocker.patch.object(settings, "anthropic_api_key", "public-key-must-not-be-used")

        await summarize(make_result(), processing_mode="nvidia_internal")

        factory.assert_called_once_with(
            api_key="nvidia-key",
            auth_token="",
            base_url="https://inference-api.nvidia.com",
            max_retries=0,
            http_client=http_client,
        )
        http_factory.assert_called_once_with(timeout=600)
        http_client.aclose.assert_awaited_once()
        assert mock_client.messages.create.call_args.kwargs["model"] == "internal-model"

    async def test_nvidia_internal_missing_key_never_constructs_client(
        self, mocker: MagicMock
    ) -> None:
        factory = mocker.patch("summarizer.AsyncAnthropic")
        mocker.patch.object(settings, "nvidia_inference_api_key", "")

        with pytest.raises(SummarizationError, match="NVIDIA_INFERENCE_API_KEY"):
            await summarize(make_result(), processing_mode="nvidia_internal")

        factory.assert_not_called()

    async def test_nvidia_internal_ignores_non_text_response_blocks(
        self, mocker: MagicMock
    ) -> None:
        response = make_claude_response(VALID_JSON)
        response.content.insert(0, MagicMock(type="thinking"))
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = response
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        mocker.patch("summarizer.nvidia_http_client", return_value=AsyncMock())
        mocker.patch.object(settings, "nvidia_inference_api_key", "nvidia-key")

        summary = await summarize(make_result(), processing_mode="nvidia_internal")

        assert summary.tldr

    async def test_local_mode_never_calls_anthropic(self, mocker: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"message": {"content": VALID_JSON}}
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        client.post.return_value = response
        client_factory = mocker.patch("summarizer.httpx.AsyncClient", return_value=client)
        anthropic = mocker.patch("summarizer.AsyncAnthropic")
        summary = await summarize(make_result(), processing_mode="local")

        assert summary.tldr
        anthropic.assert_not_called()
        assert client_factory.call_args.kwargs["trust_env"] is False

    async def test_local_request_is_reserved_before_call(self, mocker: MagicMock) -> None:
        events: list[tuple[str, int]] = []
        response = MagicMock()
        response.json.return_value = {"message": {"content": VALID_JSON}}
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None

        async def api_call(*_args: object, **_kwargs: object) -> object:
            events.append(("api", usage.local_summary_requests))
            return response

        async def persist(current: UsageStats) -> None:
            events.append(("persist", current.local_summary_requests))

        client.post.side_effect = api_call
        mocker.patch("summarizer.httpx.AsyncClient", return_value=client)
        usage = UsageStats()

        await summarize(
            make_result(),
            usage=usage,
            persist_usage=persist,
            processing_mode="local",
        )

        assert events == [("persist", 1), ("api", 1)]

    async def test_successful_call_returns_summary(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(VALID_JSON)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        result = await summarize(make_result(), processing_mode="cloud_public")
        assert result.tldr == "A great video about cycling training."
        mock_client.messages.create.assert_called_once()

    async def test_tracks_tokens_requests_and_estimated_cost(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(VALID_JSON)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        usage = UsageStats()

        await summarize(make_result(), usage=usage, processing_mode="cloud_public")

        assert usage.anthropic_requests == 1
        assert usage.anthropic_input_tokens == 100
        assert usage.anthropic_output_tokens == 50
        assert usage.estimated_cost_usd == pytest.approx(0.00105)

    async def test_persists_reservation_before_anthropic_call(self, mocker: MagicMock) -> None:
        events: list[str] = []
        mock_client = AsyncMock()

        async def api_call(**_kwargs: object) -> object:
            events.append("api")
            return make_claude_response(VALID_JSON)

        async def persist(_usage: UsageStats) -> None:
            events.append("persist")

        mock_client.messages.create.side_effect = api_call
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        await summarize(
            make_result(),
            persist_usage=persist,
            processing_mode="cloud_public",
        )

        assert events[0:2] == ["persist", "api"]

    async def test_chunk_failure_retries_only_the_failed_call(self, mocker: MagicMock) -> None:
        from anthropic import APIConnectionError

        connection_error = APIConnectionError(
            request=cast(
                Any,
                httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            )
        )
        response = make_claude_response(VALID_JSON)
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = [
            response,
            connection_error,
            response,
            response,
            response,
        ]
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        mocker.patch("summarizer.asyncio.sleep")
        mocker.patch.object(settings, "summary_chunk_chars", 10)
        mocker.patch.object(settings, "max_transcript_chars", 100)
        mocker.patch.object(settings, "max_anthropic_requests_per_job", 10)

        await summarize(
            make_result(transcript="x" * 25),
            processing_mode="cloud_public",
        )

        assert mock_client.messages.create.call_count == 5
        prompts = [
            call.kwargs["messages"][0]["content"]
            for call in mock_client.messages.create.call_args_list
        ]
        assert prompts[0].endswith("transcript chunk 1 of 3.")
        assert prompts[1].endswith("transcript chunk 2 of 3.")
        assert prompts[2].endswith("transcript chunk 2 of 3.")

    async def test_long_transcript_uses_chunk_summaries_and_synthesis(
        self, mocker: MagicMock
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(VALID_JSON)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        mocker.patch.object(settings, "summary_chunk_chars", 10)
        mocker.patch.object(settings, "max_transcript_chars", 100)
        mocker.patch.object(settings, "max_anthropic_requests_per_job", 10)
        usage = UsageStats()

        await summarize(
            make_result(transcript="x" * 25),
            usage=usage,
            processing_mode="cloud_public",
        )

        assert mock_client.messages.create.call_count == 4
        assert usage.anthropic_requests == 4

    async def test_transcript_limit_blocks_before_api_call(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        mocker.patch.object(settings, "max_transcript_chars", 10)

        with pytest.raises(UsageLimitError, match="character per-job limit"):
            await summarize(
                make_result(transcript="x" * 11),
                processing_mode="cloud_public",
            )

        mock_client.messages.create.assert_not_called()

    async def test_request_limit_blocks_chunked_summary(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        mocker.patch.object(settings, "summary_chunk_chars", 10)
        mocker.patch.object(settings, "max_transcript_chars", 100)
        mocker.patch.object(settings, "max_anthropic_requests_per_job", 3)

        with pytest.raises(UsageLimitError, match="request allowance"):
            await summarize(
                make_result(transcript="x" * 25),
                processing_mode="cloud_public",
            )

        mock_client.messages.create.assert_not_called()

    async def test_existing_request_usage_counts_toward_limit(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        usage = UsageStats(anthropic_requests=settings.max_anthropic_requests_per_job)

        with pytest.raises(UsageLimitError, match="request allowance"):
            await summarize(
                make_result(),
                usage=usage,
                processing_mode="cloud_public",
            )

        mock_client.messages.create.assert_not_called()

    async def test_cost_limit_blocks_before_api_call(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(UsageLimitError, match="cost"):
            await summarize(
                make_result(),
                cost_budget_usd=0.001,
                processing_mode="cloud_public",
            )

        mock_client.messages.create.assert_not_called()

    async def test_uses_correct_model(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(VALID_JSON)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        await summarize(make_result(), processing_mode="cloud_public")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == MODEL

    async def test_transcript_included_in_prompt(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(VALID_JSON)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        await summarize(
            make_result(transcript="My unique transcript content xyz"),
            processing_mode="cloud_public",
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_message = call_kwargs["messages"][0]["content"]
        assert "My unique transcript content xyz" in user_message

    async def test_timestamped_segments_are_included_in_prompt(self, mocker: MagicMock) -> None:
        response_json = json.dumps(
            {
                **json.loads(VALID_JSON),
                "key_moments": [{"timestamp_seconds": 65, "point": "Important detail."}],
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(response_json)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        source = make_result(
            segments=[
                TranscriptSegment(
                    start_seconds=65,
                    end_seconds=70,
                    text="Important detail.",
                )
            ]
        )

        summary = await summarize(source, processing_mode="cloud_public")

        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "[00:01:05] Important detail." in prompt
        assert summary.key_moments[0].timestamp_seconds == 65

    async def test_hallucinated_timestamp_is_removed(self, mocker: MagicMock) -> None:
        response_json = json.dumps(
            {
                **json.loads(VALID_JSON),
                "key_moments": [{"timestamp_seconds": 66, "point": "Not an actual segment start."}],
            }
        )
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(response_json)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)
        source = make_result(
            duration_seconds=0,
            segments=[TranscriptSegment(start_seconds=65, end_seconds=70, text="Actual segment.")],
        )

        summary = await summarize(source, processing_mode="cloud_public")

        assert summary.key_moments == []

    async def test_rate_limit_error_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import RateLimitError

        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = make_api_status_error(
            RateLimitError, 429, "Too many requests"
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="rate limit"):
            await summarize(make_result(), processing_mode="cloud_public")

    async def test_credit_exhaustion_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import BadRequestError

        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = make_api_status_error(
            BadRequestError, 400, "Your credit balance is too low to access the Anthropic API"
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="credit"):
            await summarize(make_result(), processing_mode="cloud_public")

    async def test_generic_bad_request_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import BadRequestError

        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = make_api_status_error(
            BadRequestError, 400, "Invalid request"
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="rejected"):
            await summarize(make_result(), processing_mode="cloud_public")

    async def test_connection_error_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import APIConnectionError

        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = APIConnectionError(
            request=cast(Any, httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="connect"):
            await summarize(make_result(), processing_mode="cloud_public")

    async def test_server_error_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import InternalServerError

        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = make_api_status_error(
            InternalServerError, 500, "Internal server error"
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="HTTP 500"):
            await summarize(make_result(), processing_mode="cloud_public")

    async def test_malformed_json_response_raises_summarization_error(
        self, mocker: MagicMock
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(
            "Sorry, I cannot summarize this content."
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="invalid JSON"):
            await summarize(make_result(), processing_mode="cloud_public")

    async def test_missing_fields_in_response_raises_summarization_error(
        self, mocker: MagicMock
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(
            json.dumps({"tldr": "ok"})  # missing key_points, tags, worth_rewatching
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError):
            await summarize(make_result(), processing_mode="cloud_public")

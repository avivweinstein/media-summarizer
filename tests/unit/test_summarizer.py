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

from exceptions import SummarizationError
from models import Summary, TranscriptResult
from summarizer import MODEL, _parse_response, summarize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_JSON = json.dumps({
    "tldr": "A great video about cycling training.",
    "key_points": ["Point one.", "Point two.", "Point three."],
    "tags": ["cycling", "fitness"],
    "worth_rewatching": True,
})


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


# ---------------------------------------------------------------------------
# summarize() — mocked Anthropic client
# ---------------------------------------------------------------------------

class TestSummarize:
    async def test_successful_call_returns_summary(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(VALID_JSON)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        result = await summarize(make_result())
        assert result.tldr == "A great video about cycling training."
        mock_client.messages.create.assert_called_once()

    async def test_uses_correct_model(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(VALID_JSON)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        await summarize(make_result())

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == MODEL

    async def test_transcript_included_in_prompt(self, mocker: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(VALID_JSON)
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        await summarize(make_result(transcript="My unique transcript content xyz"))

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_message = call_kwargs["messages"][0]["content"]
        assert "My unique transcript content xyz" in user_message

    async def test_rate_limit_error_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import RateLimitError
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = make_api_status_error(
            RateLimitError, 429, "Too many requests"
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="rate limit"):
            await summarize(make_result())

    async def test_credit_exhaustion_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import BadRequestError
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = make_api_status_error(
            BadRequestError, 400, "Your credit balance is too low to access the Anthropic API"
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="credit"):
            await summarize(make_result())

    async def test_generic_bad_request_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import BadRequestError
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = make_api_status_error(
            BadRequestError, 400, "Invalid request"
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="rejected"):
            await summarize(make_result())

    async def test_connection_error_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import APIConnectionError
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = APIConnectionError(
            request=cast(Any, httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="connect"):
            await summarize(make_result())

    async def test_server_error_raises_summarization_error(self, mocker: MagicMock) -> None:
        from anthropic import InternalServerError
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = make_api_status_error(
            InternalServerError, 500, "Internal server error"
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="HTTP 500"):
            await summarize(make_result())

    async def test_malformed_json_response_raises_summarization_error(
        self, mocker: MagicMock
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(
            "Sorry, I cannot summarize this content."
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError, match="invalid JSON"):
            await summarize(make_result())

    async def test_missing_fields_in_response_raises_summarization_error(
        self, mocker: MagicMock
    ) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = make_claude_response(
            json.dumps({"tldr": "ok"})  # missing key_points, tags, worth_rewatching
        )
        mocker.patch("summarizer.AsyncAnthropic", return_value=mock_client)

        with pytest.raises(SummarizationError):
            await summarize(make_result())

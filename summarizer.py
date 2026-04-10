"""Claude API summarizer.

Single API call returning structured JSON: tldr, key_points, tags, worth_rewatching.
JSON is validated and coerced before returning a Summary model.
"""

import json
import logging

from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    BadRequestError,
    RateLimitError,
)
from anthropic.types import TextBlock

from config import settings
from exceptions import SummarizationError
from models import Summary, TranscriptResult

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_OUTPUT_TOKENS = 1024

CANONICAL_TAGS = [
    "fitness", "cycling", "running", "lifting", "nutrition", "health",
    "finance", "investing", "tech", "ai", "productivity", "science",
    "mindset", "smart-home", "career",
]

SYSTEM_PROMPT = (
    "You are a personal knowledge assistant. Given a media transcript, produce a structured "
    "summary as JSON only — no preamble, no markdown fences.\n\n"
    "Output JSON:\n"
    "{\n"
    '  "tldr": "2-3 sentence summary",\n'
    '  "key_points": ["...", "...", "..."],\n'
    '  "tags": ["fitness", "finance"],\n'
    '  "worth_rewatching": true | false\n'
    "}\n\n"
    "Guidelines:\n"
    "- tldr: 2-3 sentences capturing the core message\n"
    "- key_points: 5-8 bullet points, each a complete sentence\n"
    "- tags: pick from the canonical list below, or add new tags if content clearly warrants it\n"
    "- worth_rewatching: true if the content meets ANY of these criteria:\n"
    "    - Dense with specific, actionable advice you'd want to revisit\n"
    "    - Contains reference material (frameworks, checklists, how-to steps) that's hard to memorize\n"
    "    - Features expert-level depth that rewards repeated viewing\n"
    "    - Has nuanced arguments that a summary alone doesn't fully capture\n"
    "  Set false if the content is entertaining but shallow, news/current events, or fully captured by the summary.\n\n"
    "Canonical tags: " + ", ".join(CANONICAL_TAGS)
)


def _build_user_prompt(result: TranscriptResult) -> str:
    lines = [
        f"Title: {result.title}",
        f"Channel/Show: {result.channel_or_show}",
        f"Source: {result.source}",
        f"Duration: {result.duration_seconds} seconds",
    ]
    if result.published_at:
        lines.append(f"Published: {result.published_at.strftime('%Y-%m-%d')}")
    lines += ["", "Transcript:", result.transcript]
    return "\n".join(lines)


def _parse_response(raw: str) -> Summary:
    """Parse and validate Claude's JSON response into a Summary.

    Strips accidental markdown fences, validates required fields, coerces types.
    Raises SummarizationError on unrecoverable parsing or validation failures.
    """
    text = raw.strip()

    # Strip markdown code fences Claude sometimes adds despite instructions
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the content between first pair of fences
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        text = inner.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SummarizationError(
            f"Claude returned invalid JSON: {e}. Raw response: {text[:300]!r}"
        ) from e

    if not isinstance(data, dict):
        raise SummarizationError(f"Expected a JSON object, got {type(data).__name__}.")

    errors: list[str] = []

    tldr = data.get("tldr", "")
    if not isinstance(tldr, str) or not tldr.strip():
        errors.append("'tldr' must be a non-empty string")
        tldr = ""

    raw_points = data.get("key_points", [])
    if not isinstance(raw_points, list):
        errors.append("'key_points' must be a list")
        key_points: list[str] = []
    else:
        key_points = [str(p) for p in raw_points]

    raw_tags = data.get("tags", [])
    if not isinstance(raw_tags, list):
        errors.append("'tags' must be a list")
        tags: list[str] = []
    else:
        tags = [str(t) for t in raw_tags]

    raw_rewatch = data.get("worth_rewatching")
    if raw_rewatch is None:
        errors.append("'worth_rewatching' is required")
        worth_rewatching = False
    elif isinstance(raw_rewatch, bool):
        worth_rewatching = raw_rewatch
    elif isinstance(raw_rewatch, str):
        worth_rewatching = raw_rewatch.lower() in ("true", "yes", "1")
    else:
        worth_rewatching = bool(raw_rewatch)

    if errors:
        raise SummarizationError(
            f"Claude response missing or invalid required fields: {'; '.join(errors)}"
        )

    return Summary(tldr=tldr, key_points=key_points, tags=tags, worth_rewatching=worth_rewatching)


async def summarize(result: TranscriptResult, job_id: str = "-") -> Summary:
    """Call Claude to summarize a transcript. Returns a validated Summary.

    Raises SummarizationError on API failures, quota exhaustion, or bad responses.
    """
    log = f"job_id={job_id} url={result.url[:60]!r} source={result.source}"
    logger.info("%s event=summarize_start transcript_chars=%d", log, len(result.transcript))

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    try:
        message = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(result)}],
        )
    except RateLimitError as e:
        raise SummarizationError(
            "Claude API rate limit reached. Please try again in a moment."
        ) from e
    except BadRequestError as e:
        msg = str(e).lower()
        if any(word in msg for word in ("credit", "billing", "balance", "quota")):
            raise SummarizationError(
                "Anthropic API credits exhausted. Please top up your account at "
                "console.anthropic.com/settings/billing."
            ) from e
        raise SummarizationError(f"Claude rejected the request: {e}") from e
    except APIConnectionError as e:
        raise SummarizationError(f"Could not connect to Claude API: {e}") from e
    except APIStatusError as e:
        raise SummarizationError(f"Claude API error (HTTP {e.status_code}): {e.message}") from e

    first_block = message.content[0]
    if not isinstance(first_block, TextBlock):
        raise SummarizationError(
            f"Unexpected response block type from Claude: {type(first_block).__name__}"
        )
    summary = _parse_response(first_block.text)

    logger.info(
        "%s event=summarize_done tags=%r worth_rewatching=%s",
        log,
        summary.tags,
        summary.worth_rewatching,
    )
    return summary

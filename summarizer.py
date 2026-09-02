"""Claude API summarizer.

One or more API calls return structured JSON: tldr, key_points, tags, worth_rewatching.
JSON is validated and coerced before returning a Summary model.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

import httpx
from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    BadRequestError,
    RateLimitError,
)
from anthropic.types import TextBlock

from config import settings
from exceptions import SummarizationError, UsageLimitError
from models import KeyMoment, Summary, TranscriptResult, UsageStats
from nvidia_inference import validate_nvidia_configuration, validated_nvidia_base_url

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_OUTPUT_TOKENS = 1024
_API_BACKOFF_SECONDS = (1, 2)


def summary_model_name(processing_mode: str) -> str:
    if processing_mode == "local":
        return f"ollama/{settings.ollama_model}"
    if processing_mode == "nvidia_internal":
        return f"nvidia-inference/{settings.nvidia_inference_model}"
    return f"anthropic/{MODEL}"


CANONICAL_TAGS = [
    "fitness",
    "cycling",
    "running",
    "lifting",
    "nutrition",
    "health",
    "finance",
    "investing",
    "tech",
    "ai",
    "productivity",
    "science",
    "mindset",
    "smart-home",
    "career",
]

SYSTEM_PROMPT = (
    "You are a personal knowledge assistant. Given a media transcript, produce a structured "
    "summary as JSON only — no preamble, no markdown fences.\n\n"
    "Output JSON:\n"
    "{\n"
    '  "tldr": "2-3 sentence summary",\n'
    '  "key_points": ["...", "...", "..."],\n'
    '  "key_moments": [{"timestamp_seconds": 123, "point": "..."}],\n'
    '  "tags": ["fitness", "finance"],\n'
    '  "worth_rewatching": true | false\n'
    "}\n\n"
    "Guidelines:\n"
    "- tldr: 2-3 sentences capturing the core message\n"
    "- key_points: 5-8 bullet points, each a complete sentence\n"
    "- key_moments: 3-5 important moments using only timestamps present in the "
    "transcript; use [] when timestamps are unavailable\n"
    "- tags: pick from the canonical list below, or add new tags if content clearly warrants it\n"
    "- worth_rewatching: true if the content meets ANY of these criteria:\n"
    "    - Dense with specific, actionable advice you'd want to revisit\n"
    "    - Contains reference material (frameworks, checklists, how-to steps) that's hard to memorize\n"
    "    - Features expert-level depth that rewards repeated viewing\n"
    "    - Has nuanced arguments that a summary alone doesn't fully capture\n"
    "  Set false if the content is entertaining but shallow, news/current events, or fully captured by the summary.\n\n"
    "Canonical tags: " + ", ".join(CANONICAL_TAGS)
)


def _format_timestamp(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timestamped_transcript(result: TranscriptResult) -> str:
    if not result.segments:
        return result.transcript
    return "\n".join(
        f"[{_format_timestamp(segment.start_seconds)}] {segment.text.strip()}"
        for segment in result.segments
    )


def _build_user_prompt(result: TranscriptResult, transcript: str | None = None) -> str:
    lines = [
        f"Title: {result.title}",
        f"Channel/Show: {result.channel_or_show}",
        f"Source: {result.source}",
        f"Duration: {result.duration_seconds} seconds",
    ]
    if result.published_at:
        lines.append(f"Published: {result.published_at.strftime('%Y-%m-%d')}")
    lines += [
        "",
        "Transcript:",
        _timestamped_transcript(result) if transcript is None else transcript,
    ]
    return "\n".join(lines)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text at whitespace when possible, with a strict character ceiling."""
    if max_chars <= 0:
        raise SummarizationError("SUMMARY_CHUNK_CHARS must be greater than zero.")
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining or not chunks:
        chunks.append(remaining)
    return chunks


def _usage_int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _anthropic_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * settings.anthropic_input_cost_per_million_usd
        + output_tokens * settings.anthropic_output_cost_per_million_usd
    ) / 1_000_000


def _local_ollama_url() -> str:
    parsed = urlparse(settings.ollama_base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise SummarizationError("Local processing requires a loopback Ollama HTTP endpoint.")
    return f"{settings.ollama_base_url.rstrip('/')}/api/chat"


async def _call_local_ollama(
    system: str,
    prompt: str,
    usage: UsageStats,
    persist_usage: Callable[[UsageStats], Awaitable[None]] | None,
) -> Summary:
    if usage.local_summary_requests >= settings.max_local_summary_requests_per_job:
        raise UsageLimitError("Configured local summary request limit reached.")
    usage.local_summary_requests += 1
    if persist_usage is not None:
        await persist_usage(usage)
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.post(
            _local_ollama_url(),
            json={
                "model": settings.ollama_model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0},
            },
        )
        response.raise_for_status()
        data = response.json()
    message = data.get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(content, str) or not content.strip():
        raise SummarizationError("The local Ollama model returned an empty summary.")
    return _parse_response(content)


async def _call_claude(
    client: AsyncAnthropic,
    *,
    model: str,
    provider_name: str,
    system: str,
    prompt: str,
    usage: UsageStats,
    cost_budget_usd: float,
    persist_usage: Callable[[UsageStats], Awaitable[None]] | None,
) -> Summary:
    estimated_input_tokens = max(1, (len(system) + len(prompt)) // 4)
    projected_cost = _anthropic_cost(estimated_input_tokens, MAX_OUTPUT_TOKENS)
    for api_attempt in range(len(_API_BACKOFF_SECONDS) + 1):
        if usage.anthropic_requests >= settings.max_anthropic_requests_per_job:
            raise UsageLimitError("Configured Anthropic per-job request limit reached.")
        if usage.estimated_cost_usd + projected_cost > cost_budget_usd:
            raise UsageLimitError(
                "Estimated summarization cost would exceed the configured per-job limit."
            )

        usage.anthropic_requests += 1
        usage.estimated_cost_usd = round(
            usage.estimated_cost_usd + projected_cost,
            6,
        )
        if persist_usage is not None:
            await persist_usage(usage)

        try:
            message = await client.messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except BadRequestError as e:
            msg = str(e).lower()
            if any(word in msg for word in ("credit", "billing", "balance", "quota")):
                raise SummarizationError(
                    f"{provider_name} credit, quota, or billing allowance is exhausted."
                ) from e
            raise SummarizationError(f"{provider_name} rejected the request.") from e
        except (RateLimitError, APIConnectionError) as e:
            if api_attempt >= len(_API_BACKOFF_SECONDS):
                if isinstance(e, RateLimitError):
                    raise SummarizationError(
                        f"{provider_name} rate limit reached. Please try again in a moment."
                    ) from e
                raise SummarizationError(f"Could not connect to {provider_name}.") from e
            await asyncio.sleep(_API_BACKOFF_SECONDS[api_attempt])
            continue
        except APIStatusError as e:
            if e.status_code >= 500 and api_attempt < len(_API_BACKOFF_SECONDS):
                await asyncio.sleep(_API_BACKOFF_SECONDS[api_attempt])
                continue
            raise SummarizationError(
                f"{provider_name} returned HTTP {e.status_code}."
            ) from e

        response_text = "".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()
        if not response_text:
            raise SummarizationError(f"{provider_name} returned no summary text.")

        input_tokens = _usage_int(getattr(message.usage, "input_tokens", 0))
        output_tokens = _usage_int(getattr(message.usage, "output_tokens", 0))
        usage.anthropic_input_tokens += input_tokens
        usage.anthropic_output_tokens += output_tokens
        actual_cost = _anthropic_cost(input_tokens, output_tokens)
        if input_tokens or output_tokens:
            usage.estimated_cost_usd = round(
                usage.estimated_cost_usd - projected_cost + actual_cost,
                6,
            )
        if persist_usage is not None:
            await persist_usage(usage)
        return _parse_response(response_text)

    raise AssertionError("Claude retry loop exhausted unexpectedly.")


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

    raw_moments = data.get("key_moments", [])
    key_moments: list[KeyMoment] = []
    if isinstance(raw_moments, list):
        for item in raw_moments:
            if not isinstance(item, dict):
                continue
            timestamp = item.get("timestamp_seconds")
            point = item.get("point")
            if isinstance(timestamp, (int, float)) and timestamp >= 0 and isinstance(point, str):
                if point.strip():
                    key_moments.append(
                        KeyMoment(timestamp_seconds=round(timestamp), point=point.strip())
                    )

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

    return Summary(
        tldr=tldr,
        key_points=key_points,
        key_moments=key_moments,
        tags=tags,
        worth_rewatching=worth_rewatching,
    )


async def summarize(
    result: TranscriptResult,
    job_id: str = "-",
    *,
    cost_budget_usd: float | None = None,
    usage: UsageStats | None = None,
    persist_usage: Callable[[UsageStats], Awaitable[None]] | None = None,
    processing_mode: str = "nvidia_internal",
) -> Summary:
    """Call Claude to summarize a transcript. Returns a validated Summary.

    Raises SummarizationError on API failures, quota exhaustion, or bad responses.
    """
    log = f"job_id={job_id} url={result.url[:60]!r} source={result.source}"
    logger.info("%s event=summarize_start transcript_chars=%d", log, len(result.transcript))

    if len(result.transcript) > settings.max_transcript_chars:
        raise UsageLimitError(
            f"Transcript has {len(result.transcript):,} characters, exceeding the configured "
            f"{settings.max_transcript_chars:,}-character per-job limit."
        )

    prompt_transcript = _timestamped_transcript(result)
    chunks = _chunk_text(prompt_transcript, settings.summary_chunk_chars)
    request_count = 1 if len(chunks) == 1 else len(chunks) + 1
    usage_tracker = usage or UsageStats()
    local_mode = processing_mode == "local"
    nvidia_internal_mode = processing_mode == "nvidia_internal"
    if local_mode:
        if (
            usage_tracker.local_summary_requests + request_count
            > settings.max_local_summary_requests_per_job
        ):
            raise UsageLimitError(
                f"Transcript requires {request_count} local summary requests, exceeding the "
                "configured per-job request allowance."
            )
        client = None
        model = settings.ollama_model
        provider_name = "Local Ollama"
    else:
        if (
            usage_tracker.anthropic_requests + request_count
            > settings.max_anthropic_requests_per_job
        ):
            raise UsageLimitError(
                f"Transcript requires {request_count} Anthropic requests, exceeding the "
                "remaining per-job request allowance."
            )
        if nvidia_internal_mode:
            try:
                validate_nvidia_configuration()
            except ValueError as error:
                raise SummarizationError(str(error)) from error
            client = AsyncAnthropic(
                api_key=settings.nvidia_inference_api_key,
                base_url=validated_nvidia_base_url(),
                max_retries=0,
            )
            model = settings.nvidia_inference_model
            provider_name = "NVIDIA Inference Hub"
        elif processing_mode == "cloud_public":
            client = AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=0)
            model = MODEL
            provider_name = "Anthropic API"
        else:
            raise SummarizationError("Unsupported processing mode.")
    budget = cost_budget_usd if cost_budget_usd is not None else settings.max_estimated_cost_usd

    async def call_summary(prompt: str) -> Summary:
        if local_mode:
            return await _call_local_ollama(
                SYSTEM_PROMPT,
                prompt,
                usage_tracker,
                persist_usage,
            )
        assert client is not None
        return await _call_claude(
            client,
            model=model,
            provider_name=provider_name,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            usage=usage_tracker,
            cost_budget_usd=budget,
            persist_usage=persist_usage,
        )

    if len(chunks) == 1:
        summary = await call_summary(_build_user_prompt(result, prompt_transcript))
    else:
        partials: list[Summary] = []
        for index, chunk in enumerate(chunks, start=1):
            partials.append(
                await call_summary(
                    _build_user_prompt(result, chunk)
                    + f"\n\nThis is transcript chunk {index} of {len(chunks)}."
                )
            )
        synthesis = json.dumps(
            [partial.model_dump() for partial in partials],
            ensure_ascii=False,
        )
        summary = await call_summary(
            _build_user_prompt(result, "")
            + "\n\nCombine these chunk summaries into one non-redundant final summary:\n"
            + synthesis
        )

    valid_timestamps = {round(segment.start_seconds) for segment in result.segments}
    summary.key_moments = [
        moment for moment in summary.key_moments if moment.timestamp_seconds in valid_timestamps
    ]

    logger.info(
        "%s event=summarize_done tags=%r worth_rewatching=%s requests=%d "
        "input_tokens=%d output_tokens=%d estimated_cost_usd=%.4f",
        log,
        summary.tags,
        summary.worth_rewatching,
        usage_tracker.anthropic_requests,
        usage_tracker.anthropic_input_tokens,
        usage_tracker.anthropic_output_tokens,
        usage_tracker.estimated_cost_usd,
    )
    return summary

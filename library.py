"""Local full-text retrieval and grounded question answering for the Obsidian archive."""

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import httpx

from models import LibraryAnswer, LibraryCitation, LibrarySearchHit

_GENERATED_DIRS = (Path("Generated/Summaries"), Path("Generated/Transcripts"))
_MAX_NOTE_BYTES = 2 * 1024 * 1024
_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "say",
    "the",
    "to",
    "what",
    "when",
    "where",
    "why",
    "with",
}


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _WORD_RE.findall(text)]


def _frontmatter_value(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$", text)
    if match is None:
        return ""
    raw = match.group(1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return str(value) if value is not None else ""


def _body_lines(text: str) -> tuple[list[str], int]:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return lines[index + 1 :], index + 1
    return lines, 0


def _search_sync(vault_path: str, query: str, limit: int) -> list[LibrarySearchHit]:
    vault = Path(vault_path).expanduser().resolve()
    if not vault.is_dir() or not (vault / ".obsidian").is_dir():
        raise ValueError("A valid Obsidian vault is required for library search.")

    raw_terms = list(dict.fromkeys(_tokens(query)))
    if not raw_terms:
        return []
    terms = [term for term in raw_terms if term not in _STOP_WORDS] or raw_terms

    hits: list[LibrarySearchHit] = []
    for relative_dir in _GENERATED_DIRS:
        directory = vault / relative_dir
        if not directory.is_dir():
            continue
        for path in directory.glob("*.md"):
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_relative_to(vault):
                continue
            if path.stat().st_size > _MAX_NOTE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            title = _frontmatter_value(text, "title") or path.stem
            lines, line_offset = _body_lines(text)
            token_counts = Counter(_tokens("\n".join(lines)))
            title_counts = Counter(_tokens(title))
            counts = [token_counts[term] for term in terms]
            if not any(counts):
                continue
            score = sum(counts) + 5 * sum(title_counts[term] for term in terms)
            line_scores = [sum(Counter(_tokens(line))[term] for term in terms) for line in lines]
            if not lines:
                continue
            best_line = max(range(len(lines)), key=line_scores.__getitem__)
            excerpt_start = max(0, best_line - 2)
            excerpt_end = min(len(lines), best_line + 3)
            excerpt = " ".join(" ".join(lines[excerpt_start:excerpt_end]).split())
            hits.append(
                LibrarySearchHit(
                    note_path=resolved.relative_to(vault).as_posix(),
                    title=title,
                    source_url=_frontmatter_value(text, "source_url") or None,
                    media_id=_frontmatter_value(text, "media_id") or None,
                    line_number=line_offset + best_line + 1,
                    excerpt=excerpt,
                    score=score,
                )
            )

    hits.sort(key=lambda hit: (-hit.score, hit.note_path))
    return hits[:limit]


async def search_library(
    vault_path: str,
    query: str,
    limit: int = 10,
) -> list[LibrarySearchHit]:
    """Search generated summaries and transcripts without reading personal notes."""
    return await asyncio.to_thread(_search_sync, vault_path, query, limit)


def _citations(hits: list[LibrarySearchHit]) -> list[LibraryCitation]:
    return [
        LibraryCitation(
            index=index,
            note_path=hit.note_path,
            title=hit.title,
            source_url=hit.source_url,
            line_number=hit.line_number,
            excerpt=hit.excerpt,
        )
        for index, hit in enumerate(hits, start=1)
    ]


async def _ask_ollama(
    question: str,
    citations: list[LibraryCitation],
    model: str,
    base_url: str,
) -> str:
    parsed_base_url = urlparse(base_url)
    if parsed_base_url.scheme != "http" or parsed_base_url.hostname not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        raise ValueError("Ollama must use a local HTTP endpoint.")

    context = "\n\n".join(f"[{item.index}] {item.title}\n{item.excerpt}" for item in citations)
    prompt = (
        "Answer only from the supplied library excerpts. Cite every factual claim using "
        "the bracketed source numbers. If the excerpts are insufficient, say so.\n\n"
        f"Question: {question}\n\nLibrary excerpts:\n{context}"
    )
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0},
            },
        )
        response.raise_for_status()
        data = response.json()
    message = data.get("message", {})
    answer = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The local Ollama model returned an empty answer.")
    return answer.strip()


async def ask_library(
    vault_path: str,
    question: str,
    *,
    limit: int,
    provider: str,
    ollama_model: str,
    ollama_base_url: str,
) -> LibraryAnswer:
    """Answer from local generated notes, with explicit source citations."""
    hits = await search_library(vault_path, question, limit)
    citations = _citations(hits)
    if not citations:
        return LibraryAnswer(
            answer="I could not find relevant material in the generated media library.",
            citations=[],
            provider=provider,
        )

    answer_provider = provider
    if provider == "ollama":
        answer = await _ask_ollama(
            question,
            citations,
            ollama_model,
            ollama_base_url,
        )
        referenced = {int(index) for index in _CITATION_RE.findall(answer)}
        valid_indices = {citation.index for citation in citations}
        if not referenced or not referenced.issubset(valid_indices):
            answer = "\n".join(
                f"{citation.excerpt} [{citation.index}]" for citation in citations
            )
            answer_provider = "extractive"
    elif provider == "extractive":
        answer = "\n".join(f"{citation.excerpt} [{citation.index}]" for citation in citations)
    else:
        raise ValueError("Library Q&A provider must be 'extractive' or 'ollama'.")

    return LibraryAnswer(answer=answer, citations=citations, provider=answer_provider)

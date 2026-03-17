# Media Summarizer

A personal media intelligence pipeline. Send YouTube or podcast URLs via WhatsApp (through OpenClaw) or HTTP, get structured AI summaries saved to Notion.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and configure

```bash
git clone <repo-url>
cd media-summarizer
cp .env.example .env
# Edit .env and fill in all API keys
```

### 3. Create venv and install dependencies

```bash
uv venv
uv pip install -e ".[dev]"
```

### 4. Start the server

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`.

---

## Install as systemd service (bear)

```bash
sudo cp media-summarizer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable media-summarizer
sudo systemctl start media-summarizer
```

View logs:

```bash
journalctl -u media-summarizer -f
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/summarize` | Submit a URL for summarization → `{ job_id }` |
| `GET`  | `/job/{job_id}` | Check job status and result |
| `GET`  | `/jobs` | List last 50 jobs |
| `GET`  | `/health` | Health check |
| `GET`  | `/` | Web UI dashboard |

### Example

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

---

## Development

```bash
# Lint
uv run ruff check .

# Type check
uv run mypy .

# Run unit tests (no API keys needed)
uv run pytest

# Run integration tests (requires real API keys in .env)
uv run pytest -m integration
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for summarization |
| `OPENAI_API_KEY` | OpenAI key for Whisper transcription |
| `NOTION_API_KEY` | Notion integration token |
| `NOTION_DATABASE_ID` | Target Notion database ID |
| `PODCAST_INDEX_API_KEY` | Podcast Index API key |
| `PODCAST_INDEX_API_SECRET` | Podcast Index API secret |
| `YOUTUBE_API_KEY` | YouTube Data API key (optional) |
| `OPENCLAW_WEBHOOK_URL` | Webhook URL for OpenClaw notifications |
| `PORT` | Server port (default: 8000) |

Get free Podcast Index API credentials at [podcastindex.org](https://podcastindex.org/developer).

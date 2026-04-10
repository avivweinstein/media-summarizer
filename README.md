# Media Summarizer

A personal media intelligence pipeline. Send YouTube or podcast URLs via the web UI, HTTP API, or WhatsApp (through OpenClaw), and get structured AI summaries saved to Notion.

## Features

- **YouTube videos** — native transcript or automatic Whisper fallback when no transcript is available
- **YouTube playlists** — auto-expands into individual video jobs
- **Podcasts** — Apple Podcasts, RSS feeds, direct MP3 URLs
- **AI summaries** — Claude generates TL;DR, key points, tags, and "worth rewatching" rating
- **Notion integration** — each summary is saved as a page in your Notion database
- **Web dashboard** — real-time job tracking with dark mode, SSE live updates
- **Job management** — cancel, delete, and auto-cleanup of old jobs
- **Concurrent processing** — async worker pool handles multiple jobs without blocking

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `ffmpeg` (for podcast audio compression and YouTube Whisper fallback)

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
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

The dashboard will be available at `http://localhost:8000`.

---

## Install as systemd user service (bear)

The service file is configured as a **user-level** systemd service (no root needed):

```bash
cp media-summarizer.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable media-summarizer
systemctl --user start media-summarizer
```

Ensure linger is enabled so the service starts on boot without an SSH session:

```bash
loginctl enable-linger $USER
```

View logs:

```bash
journalctl --user -u media-summarizer -f
```

---

## API

| Method   | Path               | Description                                           |
|----------|--------------------|-------------------------------------------------------|
| `POST`   | `/summarize`       | Submit a URL (video, podcast, or playlist) -> `{ job_id }` |
| `POST`   | `/summarize/bulk`  | Submit multiple URLs -> `{ job_ids }` |
| `GET`    | `/job/{job_id}`    | Check job status, result, and summary                 |
| `GET`    | `/jobs`            | List last 50 jobs                                     |
| `POST`   | `/job/{job_id}/cancel` | Cancel a pending/processing job                   |
| `DELETE` | `/job/{job_id}`    | Delete a single job                                   |
| `DELETE` | `/jobs/failed`     | Delete all failed jobs                                |
| `DELETE` | `/jobs/cancelled`  | Delete all cancelled jobs                             |
| `GET`    | `/jobs/stream`     | SSE stream of real-time job updates                   |
| `GET`    | `/health`          | Health check (`?deep=true` verifies API keys + DB)    |
| `GET`    | `/`                | Web UI dashboard                                      |

### Examples

```bash
# Submit a single video
curl -X POST http://bear:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'

# Submit a playlist
curl -X POST http://bear:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/playlist?list=PLxxx"}'

# Submit multiple URLs at once
curl -X POST http://bear:8000/summarize/bulk \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://youtu.be/abc", "https://youtu.be/def"]}'

# Cancel a job
curl -X POST http://bear:8000/job/{job_id}/cancel

# Delete failed jobs
curl -X DELETE http://bear:8000/jobs/failed

# Deep health check
curl http://bear:8000/health?deep=true
```

---

## Architecture

```
URL submitted via API or web UI
  |
  v
detect_source() -> "youtube" | "youtube_playlist" | "podcast"
  |                     |
  |              expand_playlist() -> fan out into individual jobs
  v
Async Worker Pool (concurrency=2)
  |
  v
Source.fetch()
  - YouTube: native transcript -> Whisper fallback if unavailable
  - Podcast: Apple/RSS -> MP3 download -> Whisper transcription
  |
  v
summarize() via Claude API -> structured JSON (tldr, key_points, tags, worth_rewatching)
  |
  v
save_to_notion() -> Notion database page with properties + body
  |
  v
Job marked done, webhook fired (if configured)
```

### Pipeline Stages

Jobs progress through stages visible in the UI:
`queued` -> `detecting` -> `transcribing` -> `summarizing` -> `saving_notion` -> `done`

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

| Variable                    | Description                                     |
|-----------------------------|-------------------------------------------------|
| `ANTHROPIC_API_KEY`         | Claude API key for summarization                |
| `OPENAI_API_KEY`            | OpenAI key for Whisper transcription            |
| `NOTION_API_KEY`            | Notion integration token                        |
| `NOTION_DATABASE_ID`        | Target Notion database ID                       |
| `PODCAST_INDEX_API_KEY`     | Podcast Index API key (reserved for future use) |
| `PODCAST_INDEX_API_SECRET`  | Podcast Index API secret                        |
| `YOUTUBE_API_KEY`           | YouTube Data API key (optional)                 |
| `OPENCLAW_WEBHOOK_URL`      | Webhook URL for OpenClaw notifications          |
| `PORT`                      | Server port (default: 8000)                     |

---

## Future Enhancements

See [ENHANCEMENTS.md](ENHANCEMENTS.md) for planned improvements including:
- Full-text search across summaries
- iOS/Android share sheet integration
- Search/filter bar in the web UI
- Additional source types (Spotify, Vimeo, Twitter Spaces)

# Media Summarizer

A personal media intelligence pipeline. Send YouTube or podcast URLs via the web UI or HTTP API, and get structured AI summaries archived in Obsidian with optional Notion publishing.

## Features

- **YouTube videos** — native transcript or automatic Whisper fallback when no transcript is available
- **YouTube playlists** — auto-expands into individual video jobs
- **Podcasts** — Apple Podcasts, RSS feeds, direct MP3 URLs
- **AI summaries** — Claude generates TL;DR, key points, tags, and "worth rewatching" rating
- **Obsidian archive** — durable local Markdown summaries with optional full transcripts
- **Notion integration** — optionally publish each summary to a Notion database
- **Web dashboard** — real-time job tracking with dark mode, SSE live updates
- **Job management** — cancel, delete, and auto-cleanup of old jobs
- **Concurrent processing** — async worker pool handles multiple jobs without blocking
- **Crash recovery and dedupe** — interrupted jobs resume after restart; equivalent submissions reuse existing work
- **Bounded API usage** — long transcripts are chunked with per-job request, size, duration, and estimated-cost limits

## Requirements

- **Python 3.11+**
- **[`uv`](https://docs.astral.sh/uv/)** — Python package manager
- **`ffmpeg`** — required for podcast audio compression and YouTube Whisper fallback

## Setup

### 1. Install system dependencies

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Clone and configure

```bash
git clone https://github.com/avivweinstein/media-summarizer.git
cd media-summarizer
cp .env.example .env
```

Edit `.env` and fill in your API keys (see [Environment Variables](#environment-variables) below for where to get each one).

### 4. Create venv and install dependencies

```bash
uv sync --extra dev --locked
```

### 5. Configure Obsidian

Create or choose an Obsidian vault, then set its absolute path in `.env`:

```dotenv
OBSIDIAN_VAULT_PATH=/Users/you/Documents/Media-Library
OBSIDIAN_RETAIN_TRANSCRIPT=true
```

The vault must already contain an `.obsidian` directory. Generated content is
written only beneath `Generated/Summaries` and `Generated/Transcripts`; existing
notes are never replaced.

> **Data boundary:** the current pipeline sends transcripts to Anthropic for
> summarization and may send downloaded audio to OpenAI for transcription. Use it
> only for public or otherwise approved media. Do not submit confidential,
> internal, restricted, or regulated material until an approved local-provider
> mode is implemented. Notion is an additional external destination when enabled.

### 6. Set up Notion (optional)

Notion publishing is disabled by default. To publish a secondary copy, set
`NOTION_ENABLED=true`, then:

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) and create a new **Internal Integration**.
2. Copy the integration token into `NOTION_API_KEY` in your `.env`.
3. Create a new Notion database with these properties:

   | Property          | Type         |
   |-------------------|--------------|
   | Title             | Title        |
   | URL               | URL          |
   | Source            | Select       |
   | Channel / Show    | Rich text    |
   | Date Added        | Date         |
   | Duration          | Number       |
   | Tags              | Multi-select |
   | Worth Rewatching  | Checkbox     |
   | TL;DR             | Rich text    |
   | Published         | Date         |
   | Thumbnail         | URL          |

4. Click **Share** on your database and invite your integration.
5. Copy the database ID from the URL (`https://notion.so/yourworkspace/<DATABASE_ID>?v=...`) into `NOTION_DATABASE_ID`.

### 7. Start the server

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in your browser. You should see the dashboard.

### 8. Verify everything works

```bash
# Quick health check
curl http://localhost:8000/health

# Deep health check — verifies all API keys and DB connectivity
curl "http://localhost:8000/health?deep=true"
```

A healthy response looks like:
```json
{"status": "ok", "db": "ok", "anthropic": "ok", "openai": "ok", "obsidian": "ok", "notion": "ok", "worker_queue_size": 0}
```

---

## Running as a background service

### macOS (launchd)

The macOS service binds to `127.0.0.1` by default so the unauthenticated API is
not exposed to the local network. It starts at login, restarts after failures,
and writes logs under `~/Library/Logs/media-summarizer/`.

```bash
uv sync --extra dev
uv run python scripts/install_macos_service.py install \
  --obsidian-vault "$HOME/Documents/Media-Library" \
  --disable-notion

# Inspect or remove the service
uv run python scripts/install_macos_service.py status
uv run python scripts/install_macos_service.py uninstall
```

The installer requires `.env` and `.venv/bin/uvicorn` to exist. The Obsidian
argument configures the vault without placing its path in `.env`.
`--disable-notion` keeps the Mac service's output local even if `.env` enables
Notion; omit it when you intentionally want both destinations. The generated
launchd configuration uses an owner-only umask so newly created runtime files
are not readable by other local users, and includes standard Homebrew paths so
`ffmpeg` remains available outside an interactive shell.

The process pauses with the Mac and resumes after wake. Pending and interrupted
jobs are recovered from SQLite and requeued automatically after a process restart.
Equivalent active submissions are deduplicated; completed static media such as a
YouTube video is reused, while RSS/show URLs can be refreshed for newer episodes.

### Linux (systemd)

To run media-summarizer as a background service that starts on boot:

### Why a systemd service?

Running as a service means:
- It starts automatically on boot (no need to SSH in and start it manually)
- It restarts automatically if it crashes
- Logs are captured by journald (viewable with `journalctl`)
- It runs without an active terminal session

### Setup

1. **Edit the service file.** Open `media-summarizer.service` and update the paths.
   The file uses `%h` which systemd expands to your home directory, so if your repo
   is at `~/media-summarizer`, it works out of the box. If it's elsewhere, update the
   `WorkingDirectory`, `ExecStart`, and `EnvironmentFile` lines.

2. **Optionally change the bind address.** The default is `127.0.0.1` (localhost only).
   To make it accessible from other devices on your network, change `--host 127.0.0.1`
   to `--host 0.0.0.0` or to a specific IP (e.g. a Tailscale IP for VPN-only access).
   See [Accessing from other devices](#accessing-from-other-devices-tailscale) below.

3. **Install and enable:**

```bash
# Copy to systemd user directory
mkdir -p ~/.config/systemd/user
cp media-summarizer.service ~/.config/systemd/user/

# Reload, enable (start on boot), and start now
systemctl --user daemon-reload
systemctl --user enable --now media-summarizer

# Enable linger so the service runs even when you're not logged in
loginctl enable-linger $USER
```

4. **Verify it's running:**

```bash
systemctl --user status media-summarizer
curl http://localhost:8000/health
```

5. **View logs:**

```bash
# Follow logs in real-time
journalctl --user -u media-summarizer -f

# Last 50 lines
journalctl --user -u media-summarizer -n 50
```

6. **Restart after pulling new code:**

```bash
cd ~/media-summarizer
git pull
uv pip install -e .
systemctl --user restart media-summarizer
```

---

## Accessing from other devices (Tailscale)

By default the server binds to `127.0.0.1` and is only reachable from the machine it runs on. To access it from your phone, laptop, or other devices, the easiest approach is [Tailscale](https://tailscale.com/) — a zero-config mesh VPN.

### Setup

1. **Install Tailscale** on the server and on every device you want to access it from ([tailscale.com/download](https://tailscale.com/download)).

2. **Find the server's Tailscale IP:**

   ```bash
   tailscale ip -4
   # Example output: 100.64.1.42
   ```

3. **Update the service file** to bind to the Tailscale IP instead of localhost:

   ```ini
   ExecStart=%h/media-summarizer/.venv/bin/uvicorn main:app --host 100.64.1.42 --port 8000
   ```

   Then reload and restart:
   ```bash
   systemctl --user daemon-reload
   systemctl --user restart media-summarizer
   ```

4. **Access from any device on your Tailscale network:**

   ```
   http://<tailscale-hostname>:8000
   ```

   Tailscale assigns a stable hostname (e.g. `my-server`) and a stable IP (e.g. `100.64.1.42`) that won't change. Both work.

### Why Tailscale?

- Your server is only reachable from devices you've authorized — no port forwarding, no exposing to the public internet
- The Tailscale IP is stable across reboots and reconnections
- Works from anywhere (home, office, mobile data) as long as Tailscale is connected
- Free for personal use (up to 100 devices)

### Alternative: bind to all interfaces

If you don't want Tailscale, you can bind to `0.0.0.0` to listen on all network interfaces. This makes the server reachable from any device on your local network, but also exposes it if your machine is on a public network. Make sure your firewall is configured appropriately.

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
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'

# Submit a playlist (auto-expands into individual jobs)
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/playlist?list=PLxxx"}'

# Submit multiple URLs at once
curl -X POST http://localhost:8000/summarize/bulk \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://youtu.be/abc", "https://youtu.be/def"]}'

# Cancel a running job
curl -X POST http://localhost:8000/job/{job_id}/cancel

# Delete all failed jobs
curl -X DELETE http://localhost:8000/jobs/failed

# Deep health check (verifies API keys)
curl "http://localhost:8000/health?deep=true"
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
save_to_obsidian() -> local Markdown summary + optional transcript
  |
  v
save_to_notion() -> optional Notion database page with properties + body
  |
  v
Job marked done, webhook fired (if configured)
```

### Pipeline Stages

Jobs progress through stages visible in the UI:
`queued` -> `detecting` -> `transcribing` -> `summarizing` -> `saving_obsidian` -> optional `saving_notion` -> `done`

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

| Variable                    | Required | Description                                     | Where to get it |
|-----------------------------|----------|-------------------------------------------------|-----------------|
| `ANTHROPIC_API_KEY`         | Yes      | Claude API key for summarization                | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `OPENAI_API_KEY`            | Yes      | OpenAI key for Whisper transcription            | [platform.openai.com](https://platform.openai.com/api-keys) |
| `OBSIDIAN_VAULT_PATH`       | No       | Local canonical archive; must contain `.obsidian` | Your vault folder |
| `OBSIDIAN_RETAIN_TRANSCRIPT` | No      | Save full transcripts to Obsidian (default: true) | — |
| `NOTION_ENABLED`            | No       | Enable optional Notion publishing (default: false) | — |
| `NOTION_API_KEY`            | If enabled | Notion integration token                      | [notion.so/my-integrations](https://www.notion.so/my-integrations) |
| `NOTION_DATABASE_ID`        | If enabled | Target Notion database ID                     | From your database URL |
| `YOUTUBE_API_KEY`           | No       | YouTube Data API key (optional metadata optimization) | [Google Cloud Console](https://console.cloud.google.com) |
| `OPENCLAW_WEBHOOK_URL`      | No       | Webhook URL for notifications                   | Your webhook endpoint |
| `WEBHOOKS_ENABLED`          | No       | Explicitly enable outbound result webhooks (default: false) | — |
| `PODCAST_INDEX_API_KEY`     | No       | Podcast Index API key (reserved for future)     | [podcastindex.org](https://podcastindex.org/developer) |
| `PODCAST_INDEX_API_SECRET`  | No       | Podcast Index API secret                        | Same as above |
| `PORT`                      | No       | Server port (default: 8000)                     | — |
| `SUMMARY_CHUNK_CHARS`       | No       | Transcript characters per summary chunk (default: 60,000) | — |
| `MAX_TRANSCRIPT_CHARS`      | No       | Hard transcript-size limit per job (default: 600,000) | — |
| `MAX_ANTHROPIC_REQUESTS_PER_JOB` | No | Anthropic request cap including retries (default: 12) | — |
| `MAX_OPENAI_REQUESTS_PER_JOB` | No     | OpenAI request cap including retries (default: 3) | — |
| `MAX_AUDIO_DURATION_SECONDS` | No      | Audio duration cap (default: 14,400 / 4 hours)  | — |
| `MAX_AUDIO_DOWNLOAD_BYTES`  | No       | Download-size cap (default: 500 MB)             | — |
| `MAX_ESTIMATED_COST_USD`    | No       | Combined estimated API spend cap per job (default: $2) | — |
| `ANTHROPIC_INPUT_COST_PER_MILLION_USD` | No | Claude input-token estimate rate (default: $3) | Anthropic pricing |
| `ANTHROPIC_OUTPUT_COST_PER_MILLION_USD` | No | Claude output-token estimate rate (default: $15) | Anthropic pricing |
| `WHISPER_COST_PER_MINUTE_USD` | No     | Whisper estimate rate (default: $0.006)         | OpenAI pricing |
| `NOTION_TEST_DATABASE_ID`   | No       | Separate Notion DB for integration tests (keeps test pages out of your real DB) | Create a blank database |

Cost estimates use configurable rates matching the pinned models: Claude Sonnet
input/output pricing and Whisper per-minute pricing. Check the official
[Anthropic pricing](https://docs.anthropic.com/en/docs/about-claude/pricing) and
[OpenAI Whisper pricing](https://developers.openai.com/api/docs/models/whisper-1)
before changing models or relying on the estimate for budgeting.

---

## Troubleshooting

**"No transcript available" for a YouTube video**
The video may not have captions. Media Summarizer will automatically fall back to Whisper transcription (downloading the audio and transcribing it). If this also fails, ensure `ffmpeg` is installed.

**Deep health check shows "not configured"**
One or more API keys are missing from your `.env` file. Check the [Environment Variables](#environment-variables) table.

**"ffmpeg not installed" error**
Whisper fallback and podcast compression require ffmpeg. Install it with `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux).

**Jobs stuck after restart**
Pending or processing jobs are requeued automatically. If one remains stuck,
inspect the service log and use the UI to cancel it before resubmitting.

**Service won't start (systemd)**
Check logs with `journalctl --user -u media-summarizer -n 50`. Common issues:
- Wrong paths in the service file
- `.env` file missing
- Python venv not created (`uv venv && uv pip install -e .`)

---

## Future Enhancements

See [ENHANCEMENTS.md](ENHANCEMENTS.md) for planned improvements including:
- Full-text search across summaries
- iOS/Android share sheet integration
- Search/filter bar in the web UI
- Additional source types (Spotify, Vimeo, Twitter Spaces)

---

## License

[MIT](LICENSE)

# Future Enhancements

Ideas and potential improvements for media-summarizer. These have been discussed
but not yet implemented — each one is a self-contained enhancement that an AI
agent or developer can pick up independently.

---

## Full-Text Search Across Summaries

**What:** Add SQLite FTS5 full-text search over job transcripts and summaries so
you can search "what was that video about zone 2 cardio?" from the web UI.

**How to implement:**
1. Create a virtual FTS5 table mirroring the `jobs` table's `result` and `summary`
   JSON text columns.
2. Add a `GET /search?q=...` endpoint that queries the FTS table and returns
   matching `JobResponse` objects ranked by relevance.
3. Add a search bar to the web UI that hits the endpoint and displays results.

**Why it's useful:** Currently the only way to find a past summary is to scroll
through the job list. With 100+ jobs this becomes unwieldy.

---

## iOS / Android Share Sheet Integration

**What:** A one-tap workflow: share a YouTube or podcast URL from any app on your
phone, and it gets submitted to media-summarizer automatically.

**How to implement (Android/Pixel):**
- Create a Tasker profile triggered by the share sheet (when URL matches
  youtube.com, podcasts.apple.com, or *.mp3).
- POST to `http://YOUR_HOST:8000/summarize` with the shared URL.
- Show a toast with the job_id.

**How to implement (iOS):**
- Create a Shortcuts automation with "Receive URL" input.
- Use the "Get Contents of URL" action to POST to the API.
- Show a notification with the result.

**Prerequisites:** Phone must be on the same network as the server (e.g. via Tailscale or local LAN).

---

## Search / Filter Bar in Web UI

**What:** Add a filter bar to the job dashboard that lets you filter by:
- Status (pending, processing, done, failed, cancelled)
- Source type (YouTube, podcast)
- Tags
- Free-text search on title

**How to implement:**
1. Add filter controls above the job list in `static/index.html`.
2. Client-side filtering on the already-loaded job list (no new API needed for
   basic filtering; add server-side for text search if FTS is implemented).

---

## More Sources

Potential source types that could be added using the existing `BaseSource` pattern:

- **Spotify podcasts** — Resolve via Spotify Web API → RSS feed → existing
  podcast pipeline. Requires Spotify API credentials.
- **Direct video files** (MP4 URLs, Vimeo) — Download audio track with yt-dlp,
  transcribe with Whisper. Most of the infrastructure exists already.
- **Twitter/X Spaces** — yt-dlp supports downloading Spaces audio. Would need
  metadata extraction from the Twitter API.
- **Web articles** — Fetch HTML, extract article text with readability, summarize.
  Different enough to warrant its own source class and summary schema.

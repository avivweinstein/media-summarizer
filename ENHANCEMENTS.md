# Future Enhancements

Remaining ideas after the Mac runtime, Obsidian archive, reliability controls,
local retrieval, and additional-source work landed.

## Ranked retrieval and grounding evaluation

Replace the current exact-token scan with an FTS5/BM25 chunk index, diversify
results by media item, and validate citations per claim. Maintain a small fixed
evaluation corpus so retrieval and grounding regressions are measurable.

## Search and filter UI

Expose the existing `/library/search` and `/library/ask` endpoints in the web UI,
along with job filters for status, source, tag, and title.

## PDF ingestion

Extract text locally from uploaded or public PDFs, preserve page-number locators,
and reuse the existing local/cloud approval boundary. Reject scanned PDFs unless
local OCR is explicitly configured.

## Mobile share sheet

Add an iOS Shortcut or Android share target only after the API has authentication
and refuses non-loopback binding without it. Remote access must not expose job
history or the media library to every LAN or tailnet peer.

## Authenticated sources

Email newsletters and social platforms may be useful later, but their account
credentials and confidential-data surfaces are substantially larger. Keep them
out of scope until access control, retention, and source-specific policy are
designed.

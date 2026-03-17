#!/usr/bin/env python3
"""Manual test for Phase 2 — YouTube transcript fetch.

Usage:
    uv run python scripts/test_phase2.py
    uv run python scripts/test_phase2.py https://www.youtube.com/watch?v=YOUR_VIDEO_ID

The server must be running:
    uv run uvicorn main:app --host 0.0.0.0 --port 8000
"""

import sys
import time

import httpx

BASE_URL = "http://localhost:8000"

# Default: a short, well-known video with a reliable transcript
DEFAULT_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" — 19s


def submit(url: str) -> str:
    r = httpx.post(f"{BASE_URL}/summarize", json={"url": url}, timeout=10)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"  submitted → job_id: {job_id}")
    return job_id


def poll(job_id: str, timeout: int = 120) -> dict:  # type: ignore[type-arg]
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(f"{BASE_URL}/job/{job_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        status = data["status"]
        print(f"  status: {status}")
        if status in ("done", "failed"):
            return data
        time.sleep(2)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def run_test(url: str) -> None:
    print(f"\n{'='*60}")
    print(f"Testing URL: {url}")
    print("="*60)

    # 1. Submit
    print("\n[1] Submitting URL...")
    job_id = submit(url)

    # 2. Poll for completion
    print("\n[2] Polling for result...")
    result = poll(job_id)

    # 3. Report
    print("\n[3] Result:")
    if result["status"] == "done":
        r = result["result"]
        print(f"  ✓ title:          {r['title']}")
        print(f"  ✓ channel:        {r['channel_or_show']}")
        print(f"  ✓ source:         {r['source']}")
        print(f"  ✓ duration:       {r['duration_seconds']}s")
        print(f"  ✓ published_at:   {r['published_at']}")
        print(f"  ✓ thumbnail_url:  {r['thumbnail_url']}")
        print(f"  ✓ transcript len: {len(r['transcript'])} chars")
        print(f"  ✓ transcript[:100]: {r['transcript'][:100]!r}")
    else:
        print(f"  ✗ FAILED: {result.get('error')}")
        sys.exit(1)


def run_rejection_test() -> None:
    print(f"\n{'='*60}")
    print("Testing URL rejection (Spotify)...")
    print("="*60)
    r = httpx.post(
        f"{BASE_URL}/summarize",
        json={"url": "https://open.spotify.com/episode/abc123"},
        timeout=10,
    )
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print(f"  ✓ Spotify rejected with 400: {r.json()['detail']}")

    r2 = httpx.post(
        f"{BASE_URL}/summarize",
        json={"url": "https://example.com/some-random-page"},
        timeout=10,
    )
    assert r2.status_code == 400, f"Expected 400, got {r2.status_code}"
    print(f"  ✓ Unknown URL rejected with 400: {r2.json()['detail']}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    try:
        httpx.get(f"{BASE_URL}/health", timeout=3).raise_for_status()
    except Exception:
        print(f"ERROR: server not reachable at {BASE_URL}")
        print("Start it with:  uv run uvicorn main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    run_rejection_test()
    run_test(url)
    print("\n✓ All Phase 2 tests passed.\n")

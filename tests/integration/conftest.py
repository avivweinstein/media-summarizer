"""Fixtures shared across integration tests."""

import os

import pytest


def _require_env(name: str) -> str:
    """Return env var value or skip the test if it's missing."""
    value = os.getenv(name, "")
    if not value:
        pytest.skip(f"Integration test requires {name} to be set")
    return value


@pytest.fixture
def notion_test_db_id() -> str:
    return _require_env("NOTION_TEST_DATABASE_ID")


@pytest.fixture
def youtube_api_key() -> str:
    return _require_env("YOUTUBE_API_KEY")

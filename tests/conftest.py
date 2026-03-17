"""Shared fixtures for all tests.

- db_path: a fresh temporary SQLite DB, initialized per-test
- More mocks will be added as phases are implemented
"""

from pathlib import Path

import pytest

import job_queue


@pytest.fixture
async def db_path(tmp_path: Path) -> str:
    """Temporary SQLite DB, initialized fresh and torn down automatically."""
    path = str(tmp_path / "test_jobs.db")
    await job_queue.init_db(db_path=path)
    return path

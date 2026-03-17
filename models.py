from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class JobStatus(StrEnum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class TranscriptResult(BaseModel):
    title: str
    source: str  # "youtube" | "podcast"
    url: str
    channel_or_show: str
    duration_seconds: int
    thumbnail_url: str | None = None
    transcript: str
    published_at: datetime | None = None


class Summary(BaseModel):
    tldr: str
    key_points: list[str]
    tags: list[str]
    worth_rewatching: bool


class Job(BaseModel):
    job_id: str
    url: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    retry_count: int = 0
    result: TranscriptResult | None = None
    summary: Summary | None = None
    notion_page_id: str | None = None
    error: str | None = None
    webhook_url: str | None = None


class SummarizeRequest(BaseModel):
    url: str
    webhook_url: str | None = None


class SummarizeResponse(BaseModel):
    job_id: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    url: str
    created_at: datetime
    updated_at: datetime
    retry_count: int = 0
    result: TranscriptResult | None = None
    summary: Summary | None = None
    notion_page_id: str | None = None
    error: str | None = None

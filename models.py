from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class JobStatus(StrEnum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class JobStage(StrEnum):
    """Fine-grained pipeline stage, shown in UI during processing."""
    queued = "queued"
    detecting = "detecting"
    transcribing = "transcribing"
    summarizing = "summarizing"
    saving_obsidian = "saving_obsidian"
    saving_notion = "saving_notion"
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
    transcription_model: str | None = None
    source_item_id: str | None = None


class Summary(BaseModel):
    tldr: str
    key_points: list[str]
    tags: list[str]
    worth_rewatching: bool


class Job(BaseModel):
    job_id: str
    url: str
    status: JobStatus
    stage: JobStage = JobStage.queued
    created_at: datetime
    updated_at: datetime
    retry_count: int = 0
    result: TranscriptResult | None = None
    summary: Summary | None = None
    notion_page_id: str | None = None
    notion_error: str | None = None
    obsidian_note_path: str | None = None
    error: str | None = None
    webhook_url: str | None = None
    parent_job_id: str | None = None  # set when this job was spawned from a playlist/bulk
    dedupe_key: str | None = None


class SummarizeRequest(BaseModel):
    url: str
    webhook_url: str | None = None


class SummarizeResponse(BaseModel):
    job_id: str


class BulkSummarizeRequest(BaseModel):
    """Submit multiple URLs or a playlist URL."""
    urls: list[str]
    webhook_url: str | None = None


class BulkSummarizeResponse(BaseModel):
    job_ids: list[str]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage = JobStage.queued
    url: str
    created_at: datetime
    updated_at: datetime
    retry_count: int = 0
    result: TranscriptResult | None = None
    summary: Summary | None = None
    notion_page_id: str | None = None
    notion_error: str | None = None
    obsidian_note_path: str | None = None
    error: str | None = None
    parent_job_id: str | None = None

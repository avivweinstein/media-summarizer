from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


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


class UsageStats(BaseModel):
    anthropic_requests: int = 0
    anthropic_input_tokens: int = 0
    anthropic_output_tokens: int = 0
    openai_requests: int = 0
    openai_audio_seconds: float = 0
    local_summary_requests: int = 0
    estimated_cost_usd: float = 0


class TranscriptSegment(BaseModel):
    start_seconds: float
    end_seconds: float
    text: str


class TranscriptionOutput(BaseModel):
    text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)


class TranscriptResult(BaseModel):
    title: str
    source: str  # e.g. "youtube" | "podcast" | "twitter" | "article"
    url: str
    channel_or_show: str
    duration_seconds: int
    thumbnail_url: str | None = None
    transcript: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    published_at: datetime | None = None
    transcription_model: str | None = None
    source_item_id: str | None = None


class KeyMoment(BaseModel):
    timestamp_seconds: int
    point: str


class Summary(BaseModel):
    tldr: str
    key_points: list[str]
    key_moments: list[KeyMoment] = Field(default_factory=list)
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
    webhook_urls: list[str] = Field(default_factory=list)
    parent_job_id: str | None = None  # set when this job was spawned from a playlist/bulk
    dedupe_key: str | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    interrupted: bool = False
    interruption_count: int = 0
    processing_mode: str = "cloud_public"
    external_processing_approved: bool = False


class SummarizeRequest(BaseModel):
    url: str
    webhook_url: str | None = None
    external_processing_approved: bool = False


class SummarizeResponse(BaseModel):
    job_id: str


class BulkSummarizeRequest(BaseModel):
    """Submit multiple URLs or a playlist URL."""

    urls: list[str]
    webhook_url: str | None = None
    external_processing_approved: bool = False


class BulkSummarizeResponse(BaseModel):
    job_ids: list[str]


class LibrarySearchHit(BaseModel):
    note_path: str
    title: str
    source_url: str | None = None
    media_id: str | None = None
    line_number: int
    excerpt: str
    score: int


class LibraryCitation(BaseModel):
    index: int
    note_path: str
    title: str
    source_url: str | None = None
    line_number: int
    excerpt: str


class LibraryAskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)


class LibraryAnswer(BaseModel):
    answer: str
    citations: list[LibraryCitation]
    provider: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage = JobStage.queued
    url: str
    created_at: datetime
    updated_at: datetime
    retry_count: int = 0
    interruption_count: int = 0
    result: TranscriptResult | None = None
    summary: Summary | None = None
    notion_page_id: str | None = None
    notion_error: str | None = None
    obsidian_note_path: str | None = None
    error: str | None = None
    parent_job_id: str | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    processing_mode: str = "cloud_public"

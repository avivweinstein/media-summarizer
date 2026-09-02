from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    nvidia_inference_api_key: str = ""
    nvidia_inference_base_url: str = "https://inference-api.nvidia.com"
    nvidia_inference_model: str = "us/azure/anthropic/eccn-claude-sonnet-5"
    notion_api_key: str = ""
    notion_database_id: str = ""
    notion_enabled: bool = False
    obsidian_vault_path: str = ""
    obsidian_retain_transcript: bool = True
    db_retain_transcript: bool = False
    job_db_path: str = "~/Library/Application Support/media-summarizer/jobs.db"
    backup_dir: str = "~/Library/Application Support/media-summarizer/backups"
    podcast_index_api_key: str = ""
    podcast_index_api_secret: str = ""
    youtube_api_key: str = ""
    openclaw_webhook_url: str = ""
    webhooks_enabled: bool = False
    library_qa_provider: Literal["extractive", "ollama"] = "extractive"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    processing_mode: Literal["nvidia_internal", "cloud_public", "local"] = "nvidia_internal"
    upload_dir: str = "~/Library/Application Support/media-summarizer/uploads"
    local_whisper_executable: str = "whisper-cli"
    local_whisper_model: str = ""
    local_ffmpeg_timeout_seconds: int = Field(default=600, gt=0)
    local_whisper_timeout_seconds: int = Field(default=14_400, gt=0)
    port: int = 8000
    summary_chunk_chars: int = Field(default=60_000, gt=0)
    max_transcript_chars: int = Field(default=600_000, gt=0)
    max_anthropic_requests_per_job: int = Field(default=12, gt=0)
    max_local_summary_requests_per_job: int = Field(default=12, gt=0)
    max_openai_requests_per_job: int = Field(default=3, gt=0)
    max_audio_duration_seconds: int = Field(default=14_400, gt=0)
    max_audio_download_bytes: int = Field(default=500_000_000, gt=0)
    source_fetch_timeout_seconds: int = Field(default=120, gt=0)
    max_article_download_bytes: int = Field(default=5_000_000, gt=0)
    max_estimated_cost_usd: float = Field(default=2.0, gt=0)
    anthropic_input_cost_per_million_usd: float = Field(default=3.0, ge=0)
    anthropic_output_cost_per_million_usd: float = Field(default=15.0, ge=0)
    whisper_cost_per_minute_usd: float = Field(default=0.006, ge=0)


settings = Settings()

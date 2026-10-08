from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    notion_api_key: str = ""
    notion_database_id: str = ""
    notion_enabled: bool = False
    obsidian_vault_path: str = ""
    obsidian_retain_transcript: bool = True
    podcast_index_api_key: str = ""
    podcast_index_api_secret: str = ""
    youtube_api_key: str = ""
    openclaw_webhook_url: str = ""
    port: int = 8000


settings = Settings()

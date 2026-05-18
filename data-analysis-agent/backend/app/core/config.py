from functools import lru_cache

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Data Analysis Agent API"
    database_url: str = "sqlite:///./data_analysis_agent.db"
    storage_dir: str = ".data"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1"
    backend_cors_origins: list[AnyHttpUrl] = Field(default_factory=lambda: ["http://localhost:5173"])

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str) and not value.startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]

        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

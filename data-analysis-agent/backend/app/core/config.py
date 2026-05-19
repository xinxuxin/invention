from functools import lru_cache

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Data Analysis Agent API"
    database_url: str = "sqlite:///./data_analysis_agent.db"
    storage_dir: str = ".data"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1"
    agent_model_mode: str = "openai"
    agent_mode: str | None = None
    fake_agent_mode: bool = False
    agent_max_steps: int = 6
    agent_max_retries: int = 3
    verifier_mode: str = "hybrid"
    llm_verifier_enabled: bool = True
    llm_verifier_model: str = "gpt-4.1-mini"
    llm_verifier_timeout_seconds: int = 8
    llm_verifier_max_tokens: int = 700
    llm_verifier_max_input_chars: int = 12_000
    llm_verifier_min_confidence: float = 0.70
    llm_verifier_fail_open: bool = True
    llm_verifier_hard_rule_authority: bool = False
    llm_verifier_policy: str = "selective"
    llm_verifier_complexity_threshold: int = 3
    llm_verifier_conceptual_enabled: bool = True
    llm_verifier_repeat_retry_enabled: bool = True
    llm_verifier_max_calls_per_turn: int = 1
    show_verifier_debug_trace: bool = False
    verifier_time_budget_per_turn_seconds: int = 8
    verifier_skip_llm_after_step: int = 4
    frontend_inline_table_max_rows: int = 50
    frontend_inline_table_max_columns: int = 30
    frontend_cell_max_chars: int = 200
    frontend_chart_max_points: int = 500
    backend_cors_origins: list[AnyHttpUrl] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

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


def resolved_agent_mode(settings: Settings | None = None) -> str:
    current = settings or get_settings()
    if current.fake_agent_mode:
        return "fake"
    if (current.agent_mode or "").lower() == "fake":
        return "fake"
    if current.agent_model_mode.lower() == "fake":
        return "fake"
    return "real"

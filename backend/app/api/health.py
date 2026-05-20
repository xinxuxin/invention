from fastapi import APIRouter

from app.core.config import get_settings, resolved_agent_mode
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="data-analysis-agent-api")


@router.get("/health/config")
async def health_config() -> dict[str, object]:
    settings = get_settings()
    agent_mode = resolved_agent_mode(settings)
    return {
        "agent_mode": agent_mode,
        "agent_model_mode": settings.agent_model_mode,
        "fake_agent_mode": agent_mode == "fake",
        "llm_verifier_enabled": settings.llm_verifier_enabled,
        "verifier_mode": settings.verifier_mode,
        "llm_verifier_model": settings.llm_verifier_model,
        "llm_verifier_timeout_seconds": settings.llm_verifier_timeout_seconds,
        "python_execution_timeout_seconds": settings.python_execution_timeout_seconds,
        "mutation_persist_timeout_seconds": settings.mutation_persist_timeout_seconds,
        "agent_max_steps": settings.agent_max_steps,
        "agent_max_retries": settings.agent_max_retries,
        "llm_verifier_policy": settings.llm_verifier_policy,
        "llm_verifier_complexity_threshold": settings.llm_verifier_complexity_threshold,
        "verifier_skip_llm_after_step": settings.verifier_skip_llm_after_step,
        "has_openai_api_key": bool(settings.openai_api_key),
    }

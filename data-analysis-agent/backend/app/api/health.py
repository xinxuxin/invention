from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="data-analysis-agent-api")


@router.get("/health/config")
async def health_config() -> dict[str, object]:
    settings = get_settings()
    return {
        "llm_verifier_enabled": settings.llm_verifier_enabled,
        "verifier_mode": settings.verifier_mode,
        "llm_verifier_model": settings.llm_verifier_model,
        "llm_verifier_timeout_seconds": settings.llm_verifier_timeout_seconds,
        "verifier_skip_llm_after_step": settings.verifier_skip_llm_after_step,
        "has_openai_api_key": bool(settings.openai_api_key),
    }

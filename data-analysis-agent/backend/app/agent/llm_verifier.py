from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from app.agent.types import LLMVerificationResult, VerificationResult
from app.core.config import Settings, get_settings
from app.runtime.python_executor import ExecutionArtifact, ExecutionResult

LLM_VERIFIER_SYSTEM_PROMPT = """You are a verifier for a data-analysis coding agent. Your job is to check whether the latest execution and artifacts satisfy the user's request. You are not the data analyst. Do not invent data. Use only provided execution summaries, result previews, artifacts, and state metadata. Prefer deterministic evidence. If requested artifacts are missing, say so. If the answer can be finalized, provide concise final_answer_guidance. Return strict JSON only."""


class LLMVerifier:
    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client

    def verify_if_allowed(
        self,
        *,
        user_message: str,
        context: dict[str, Any],
        execution_result: ExecutionResult | None,
        artifacts: list[ExecutionArtifact],
        state_changed: bool,
        latest_code: str | None,
        deterministic_result: VerificationResult,
        current_step: int,
        turn_started_at: float,
    ) -> tuple[LLMVerificationResult | None, str | None]:
        if not self._allowed(deterministic_result, current_step, turn_started_at):
            return None, self._skip_reason(deterministic_result, current_step, turn_started_at)

        try:
            return self.verify(
                user_message=user_message,
                context=context,
                execution_result=execution_result,
                artifacts=artifacts,
                state_changed=state_changed,
                latest_code=latest_code,
                deterministic_result=deterministic_result,
            ), None
        except Exception as exc:
            if self.settings.llm_verifier_fail_open:
                return None, f"LLM verifier unavailable; using deterministic verifier. ({type(exc).__name__})"
            raise

    def verify(
        self,
        *,
        user_message: str,
        context: dict[str, Any],
        execution_result: ExecutionResult | None,
        artifacts: list[ExecutionArtifact],
        state_changed: bool,
        latest_code: str | None,
        deterministic_result: VerificationResult,
    ) -> LLMVerificationResult:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        if self.client is None:
            self.client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.llm_verifier_timeout_seconds,
            )

        payload = self._payload(
            user_message=user_message,
            context=context,
            execution_result=execution_result,
            artifacts=artifacts,
            state_changed=state_changed,
            latest_code=latest_code,
            deterministic_result=deterministic_result,
        )
        response = self.client.responses.create(
            model=self.settings.llm_verifier_model,
            instructions=LLM_VERIFIER_SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": _truncate_json(payload, self.settings.llm_verifier_max_input_chars),
                }
            ],
            max_output_tokens=self.settings.llm_verifier_max_tokens,
        )
        text = getattr(response, "output_text", None) or ""
        parsed = json.loads(_strip_json_fence(text))
        return LLMVerificationResult.model_validate(parsed)

    def _allowed(
        self,
        deterministic_result: VerificationResult,
        current_step: int,
        turn_started_at: float,
    ) -> bool:
        mode = self.settings.verifier_mode.lower()
        if not self.settings.llm_verifier_enabled or mode not in {"llm", "hybrid"}:
            return False
        if not self.settings.openai_api_key:
            return False
        if deterministic_result.hard_fail:
            return False
        if deterministic_result.severity == "retry" and _deterministic_retry_is_authoritative(deterministic_result):
            return False
        if current_step > self.settings.verifier_skip_llm_after_step:
            return False
        return time.monotonic() - turn_started_at <= self.settings.verifier_time_budget_per_turn_seconds

    def _skip_reason(
        self,
        deterministic_result: VerificationResult,
        current_step: int,
        turn_started_at: float,
    ) -> str | None:
        mode = self.settings.verifier_mode.lower()
        if mode == "deterministic" or not self.settings.llm_verifier_enabled:
            return None
        if not self.settings.openai_api_key:
            return "LLM verifier unavailable; using deterministic verifier. (missing API key)"
        if deterministic_result.hard_fail or (
            deterministic_result.severity == "retry" and _deterministic_retry_is_authoritative(deterministic_result)
        ):
            return None
        if current_step > self.settings.verifier_skip_llm_after_step:
            return "Skipping LLM verifier to keep response fast."
        if time.monotonic() - turn_started_at > self.settings.verifier_time_budget_per_turn_seconds:
            return "Skipping LLM verifier to keep response fast."
        return None

    def _payload(
        self,
        *,
        user_message: str,
        context: dict[str, Any],
        execution_result: ExecutionResult | None,
        artifacts: list[ExecutionArtifact],
        state_changed: bool,
        latest_code: str | None,
        deterministic_result: VerificationResult,
    ) -> dict[str, Any]:
        return {
            "user_message": user_message,
            "context": {
                "active_dataset_key": context.get("active_dataset_key"),
                "dataset_keys": context.get("dataset_keys"),
                "active_branch": context.get("active_branch"),
                "datasets": context.get("datasets", [])[:5],
            },
            "latest_execution": {
                "ok": execution_result.ok if execution_result else None,
                "stdout": _compact(execution_result.stdout if execution_result else "", 1200),
                "stderr": _compact(execution_result.stderr if execution_result else "", 1200),
                "traceback": _compact(execution_result.traceback or "", 1600) if execution_result else None,
                "result_preview": execution_result.result_preview if execution_result else None,
            },
            "artifacts": [
                {
                    "id": artifact.id,
                    "kind": artifact.kind,
                    "title": artifact.title or artifact.name,
                    "columns": artifact.columns[:30],
                    "chart_spec": artifact.chart_spec,
                    "metadata": artifact.metadata,
                }
                for artifact in artifacts
            ],
            "state_changed": state_changed,
            "latest_code_excerpt": _compact(latest_code or "", 2000),
            "deterministic_verifier": deterministic_result.model_dump(mode="json"),
        }


def _deterministic_retry_is_authoritative(result: VerificationResult) -> bool:
    text = " ".join([*result.reasons, result.retry_instruction or ""]).lower()
    return any(marker in text for marker in ("table artifact", "chart artifact", "csv artifact", "wrapper columns", "state changed"))


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()


def _truncate_json(payload: dict[str, Any], limit: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n... truncated ..."


def _compact(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit].rstrip()}\n... truncated ..."

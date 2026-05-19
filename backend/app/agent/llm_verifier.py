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
        force: bool = False,
        calls_this_turn: int = 0,
    ) -> tuple[LLMVerificationResult | None, str | None]:
        complexity_score = score_task_complexity(
            user_message,
            latest_execution_result=execution_result,
            artifacts=artifacts,
            deterministic_result=deterministic_result,
        )
        if not self._allowed(
            user_message,
            deterministic_result,
            current_step,
            turn_started_at,
            force=force,
            calls_this_turn=calls_this_turn,
            complexity_score=complexity_score,
        ):
            return None, self._skip_reason(
                user_message,
                deterministic_result,
                current_step,
                turn_started_at,
                force=force,
                calls_this_turn=calls_this_turn,
                complexity_score=complexity_score,
            )

        try:
            result = self.verify(
                user_message=user_message,
                context=context,
                execution_result=execution_result,
                artifacts=artifacts,
                state_changed=state_changed,
                latest_code=latest_code,
                deterministic_result=deterministic_result,
            )
            return result, "Semantic verifier reviewed the result."
        except Exception as exc:
            if self.settings.llm_verifier_fail_open:
                return None, self._fallback_trace(exc)
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
        parsed = _parse_llm_json(text)
        parsed.setdefault("passed", False)
        parsed.setdefault("confidence", 0.0)
        parsed.setdefault("missing_requirements", [])
        parsed.setdefault("hallucination_risk", "medium")
        parsed.setdefault("retry_instruction", None)
        parsed.setdefault("final_answer_guidance", None)
        parsed.setdefault("should_finalize", False)
        parsed.setdefault("reasons", [])
        return LLMVerificationResult.model_validate(parsed)

    def _allowed(
        self,
        user_message: str,
        deterministic_result: VerificationResult,
        current_step: int,
        turn_started_at: float,
        force: bool = False,
        calls_this_turn: int = 0,
        complexity_score: int = 0,
    ) -> bool:
        mode = self.settings.verifier_mode.lower()
        if not self.settings.llm_verifier_enabled or mode not in {"llm", "hybrid"}:
            return False
        if self.settings.llm_verifier_policy.lower() == "never":
            return False
        if calls_this_turn >= self.settings.llm_verifier_max_calls_per_turn:
            return False
        if not self.settings.openai_api_key:
            return False
        if deterministic_result.hard_fail:
            return False
        if deterministic_result.severity == "retry" and _deterministic_retry_is_authoritative(deterministic_result) and not force:
            return False
        if current_step > self.settings.verifier_skip_llm_after_step and not force and not _semantic_verification_priority(user_message):
            return False
        if self.settings.llm_verifier_policy.lower() == "always":
            return True
        if force:
            return self.settings.llm_verifier_repeat_retry_enabled
        if complexity_score >= self.settings.llm_verifier_complexity_threshold:
            return True
        if self.settings.llm_verifier_conceptual_enabled and _semantic_verification_priority(user_message):
            return True
        if deterministic_result.severity in {"retry", "finalize_with_warning"} and not _deterministic_retry_is_authoritative(deterministic_result):
            return True
        return False

    def _skip_reason(
        self,
        user_message: str,
        deterministic_result: VerificationResult,
        current_step: int,
        turn_started_at: float,
        force: bool = False,
        calls_this_turn: int = 0,
        complexity_score: int = 0,
    ) -> str | None:
        if not self.settings.show_verifier_debug_trace:
            return None
        mode = self.settings.verifier_mode.lower()
        if mode == "deterministic" or not self.settings.llm_verifier_enabled:
            return None
        if self.settings.llm_verifier_policy.lower() == "never":
            return "LLM verifier skipped by policy."
        if calls_this_turn >= self.settings.llm_verifier_max_calls_per_turn:
            return "LLM verifier skipped after reaching the per-turn call limit."
        if not self.settings.openai_api_key:
            return self._fallback_trace("missing API key")
        if deterministic_result.hard_fail or (
            deterministic_result.severity == "retry" and _deterministic_retry_is_authoritative(deterministic_result)
        ):
            return None
        if current_step > self.settings.verifier_skip_llm_after_step and not force:
            return "Skipping LLM verifier to keep response fast."
        if (
            time.monotonic() - turn_started_at > self.settings.verifier_time_budget_per_turn_seconds
            and not _semantic_verification_priority(user_message)
        ):
            return "Skipping LLM verifier to keep response fast."
        return None

    def _fallback_trace(self, reason: object) -> str:
        if self.settings.show_verifier_debug_trace:
            if isinstance(reason, BaseException):
                detail = type(reason).__name__
            else:
                detail = str(reason)
            return f"Verifier completed using deterministic checks. ({detail})"
        return "Verifier completed using deterministic checks."

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


def _semantic_verification_priority(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(
        marker in lowered
        for marker in (
            "schema",
            "scalar",
            "date fields",
            "list-like",
            "what is in this file",
            "what is this",
            "what's in this file",
            "summarize",
            "explain",
            "what does this dataset represent",
            "important fields",
            "explain the structure",
            "top-level keys",
            "object types",
            "custom class",
            "mixed top-level",
            "compare datasets",
            "compare the uploaded datasets",
            "identify possible join keys",
            "data quality",
            "bad records",
            "duplicates",
            "invalid dates",
        )
    )


def score_task_complexity(
    user_message: str,
    *,
    latest_execution_result: ExecutionResult | None,
    artifacts: list[ExecutionArtifact],
    deterministic_result: VerificationResult,
) -> int:
    lowered = user_message.lower()
    score = 0
    if any(marker in lowered for marker in (" and ", " also ", " then ", " as well as ", "table and chart", "export and summarize")):
        score += 2
    if "table" in lowered and any(marker in lowered for marker in ("chart", "graph", "plot", "visualize")):
        score += 2
    if _semantic_verification_priority(user_message):
        score += 2
    if any(marker in lowered for marker in ("compare datasets", "both datasets", "join", "original vs", "dataset a and dataset b")):
        score += 2
    if any(marker in lowered for marker in ("compare current branch", "rollback", "fork", "what changed")):
        score += 2
    if deterministic_result.severity in {"retry", "finalize_with_warning"} and not _deterministic_retry_is_authoritative(deterministic_result):
        score += 2
    if artifacts and deterministic_result.metadata.get("intent_uncertain"):
        score += 2
    if deterministic_result.severity == "retry":
        score += 2
    preview = latest_execution_result.result_preview if latest_execution_result is not None else None
    if isinstance(preview, (dict, list)) and len(json.dumps(preview, default=str)) > 2500:
        score += 1
    return score


def _parse_llm_json(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = json.loads(_extract_first_json_object(cleaned))
    if not isinstance(parsed, dict):
        raise ValueError("LLM verifier response must be a JSON object")
    return parsed


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.strip()


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        character = text[index]
        if escape:
            escape = False
            continue
        if character == "\\":
            escape = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise json.JSONDecodeError("Unterminated JSON object", text, start)


def _truncate_json(payload: dict[str, Any], limit: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n... truncated ..."


def _compact(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit].rstrip()}\n... truncated ..."

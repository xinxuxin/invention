from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str


class ChatStreamRequest(BaseModel):
    message: str = Field(min_length=1)
    active_dataset_id: str | None = None
    branch_name: str = "main"
    conversation_history: list[ChatHistoryMessage] = Field(default_factory=list)
    confirmed: bool = False


class AgentAction(BaseModel):
    kind: Literal["execute_python", "final_answer", "request_confirmation"]
    code: str | None = None
    mutates_state: bool = False
    mutation_summary: str | None = None
    answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    id: str
    type: Literal["table", "chart", "csv", "json"] | str
    title: str
    description: str | None = None
    columns: list[Any] | None = None
    rows: list[Any] | None = None
    chart_spec: dict[str, Any] | None = None
    download_url: str | None = None


class ExecutionResult(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    traceback: str | None = None
    result_preview: Any | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    state_changed: bool = False
    mutation_summary: str | None = None
    created_version_id: str | None = None


class VerificationResult(BaseModel):
    passed: bool
    severity: Literal["pass", "retry", "finalize_with_warning", "fail"]
    reasons: list[str] = Field(default_factory=list)
    retry_instruction: str | None = None
    should_finalize: bool = False
    confidence: float = 1.0
    source: Literal["deterministic", "llm", "hybrid"] = "deterministic"
    hard_fail: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMVerificationResult(BaseModel):
    passed: bool
    confidence: float = 0.0
    missing_requirements: list[str] = Field(default_factory=list)
    hallucination_risk: Literal["low", "medium", "high"] = "medium"
    retry_instruction: str | None = None
    final_answer_guidance: str | None = None
    should_finalize: bool = False
    reasons: list[str] = Field(default_factory=list)


class ComposedAnswer(BaseModel):
    markdown: str
    highlights: list[dict[str, Any]] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    state_changed: bool = False
    artifact_ids: list[str] = Field(default_factory=list)


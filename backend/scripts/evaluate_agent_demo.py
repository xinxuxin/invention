from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORBIDDEN_COLUMNS = {
    "__dict__",
    "__pydantic_extra__",
    "__pydantic_fields_set__",
    "__pydantic_private__",
}

GLOBAL_FORBIDDEN_STRINGS = [
    "reached internal step budget",
    "reached its internal step budget",
    "could not complete",
    "signal is aborted",
    "NameError",
    "No module named",
    "globals is not defined",
    "locals is not defined",
    "Chart spec must include data",
    "__pydantic",
    "{'type':",
    "'attrs'",
    "Fake agent chart",
]

GENERATED_DATASET_FILENAMES = {
    "nested_customer_events": "nested_customer_events.pkl",
    "mixed_dataframe_numpy_bundle": "mixed_dataframe_numpy_bundle.pkl",
    "custom_sensor_fleet": "custom_sensor_fleet.pkl",
    "mixed_top_level_collection": "mixed_top_level_collection.pkl",
}


@dataclass
class EvalQuestion:
    id: str
    prompt: str
    checks: list[str | dict[str, Any]] = field(default_factory=list)
    warnings: list[str | dict[str, Any]] = field(default_factory=list)
    dataset_key: str | None = None
    special: str | None = None


@dataclass
class DatasetUpload:
    path: Path
    dataset_id: str
    filename: str
    key: str | None = None


@dataclass
class QuestionResult:
    id: str
    question: str
    status: str
    failure_reasons: list[str]
    warning_reasons: list[str]
    final_answer: str
    artifacts: list[dict[str, Any]]
    execute_python_calls: int
    code_failed_count: int
    verifier_retries: int
    llm_verifier_called: bool
    llm_verifier_skipped: bool
    llm_verifier_fallback_used: bool
    llm_verifier_skip_reason: str | None
    state_changed: bool
    confirmation_required: bool
    raw_events_path: str
    screenshot_path: str | None = None
    notes: list[str] = field(default_factory=list)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Data Analysis Agent evaluation suites and generate human-reviewable reports."
    )
    parser.add_argument("--quick", action="store_true", help="Run a shorter subset of the selected suite.")
    parser.add_argument("--full", action="store_true", help="Run the full selected suite.")
    parser.add_argument("--suite", default="softbank_core", help="Suite name, e.g. softbank_core or generated_all.")
    parser.add_argument(
        "--dataset",
        default=os.environ.get(
            "E2E_SOFTBANK_PKL_PATH",
            "/Users/macbook/Desktop/softbank_group_patent_portfolio_metadata.pkl",
        ),
    )
    parser.add_argument("--dataset-dir", default=os.environ.get("AGENT_TEST_DATASET_DIR"))
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--frontend-url", default="http://localhost:5173")
    parser.add_argument("--out", "--out-dir", dest="out", default="backend/eval_reports/latest")
    parser.add_argument("--approve-mutations", action="store_true")
    parser.add_argument("--use-real-agent", action="store_true", help="Require the running backend to be in real mode.")
    parser.add_argument("--use-fake-agent", action="store_true", help="Require the running backend to be in fake mode.")
    parser.add_argument("--browser-smoke", action="store_true")
    parser.add_argument("--include-screenshots", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--restart-backend-check", action="store_true")
    parser.add_argument("--debug-trace", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    frontend_url = args.frontend_url.rstrip("/")
    out_dir = Path(args.out).expanduser()
    raw_events_dir = out_dir / "raw_events"
    screenshots_dir = out_dir / "screenshots"
    raw_events_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    health_config = _get_health_config(base_url)
    mode = str(health_config.get("agent_mode") or "unknown")
    if args.use_real_agent and mode != "real":
        print(f"Backend agent mode is {mode}, expected real.", file=sys.stderr)
        return 2
    if args.use_fake_agent and mode != "fake":
        print(f"Backend agent mode is {mode}, expected fake.", file=sys.stderr)
        return 2

    suite_names = _expand_suite_alias(args.suite)
    dataset_paths = _resolve_dataset_paths(args, suite_names)
    session_id = args.session_id or _create_session(base_url, f"Eval {args.suite} {int(time.time())}")["id"]
    uploads = _upload_required_datasets(base_url, session_id, dataset_paths, existing_session=bool(args.session_id))
    active_dataset_id = uploads[0].dataset_id if uploads else _active_dataset_id(base_url, session_id)

    browser = BrowserSmoke(
        enabled=args.browser_smoke or args.include_screenshots,
        frontend_url=frontend_url,
        session_id=session_id,
        screenshots_dir=screenshots_dir,
    )
    browser_warning = browser.start()

    all_questions = _questions_for_suites(suite_names)
    if args.quick:
        all_questions = _quick_subset(all_questions, args.suite)

    results: list[QuestionResult] = []
    transcript: list[tuple[str, str]] = []
    uploaded_by_suite = {upload.path.name: upload for upload in uploads}

    for index, question in enumerate(all_questions, start=1):
        question_id = f"q{index:02d}_{_safe_slug(question.id)}"
        if question.special:
            result = _run_special_check(
                base_url=base_url,
                session_id=session_id,
                question=question,
                question_id=question_id,
                raw_events_dir=raw_events_dir,
                browser=browser,
                debug_trace=args.debug_trace,
                restart_backend_check=args.restart_backend_check,
            )
            results.append(result)
            continue

        dataset_id = active_dataset_id
        if question.dataset_key:
            upload = uploaded_by_suite.get(f"{question.dataset_key}.pkl")
            if upload:
                dataset_id = upload.dataset_id
                _activate_dataset(base_url, session_id, dataset_id)

        events = _chat(
            base_url,
            session_id,
            dataset_id,
            question.prompt,
            approve_mutations=args.approve_mutations,
        )
        raw_path = raw_events_dir / f"{question_id}.json"
        raw_path.write_text(json.dumps(events, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        final = _last_event(events, "final_answer")
        artifacts = [event["artifact"] for event in events if event.get("type") == "artifact_created" and event.get("artifact")]
        artifact_contents = {
            str(artifact["id"]): _artifact_content(base_url, session_id, str(artifact["id"]))
            for artifact in artifacts
            if artifact.get("id")
        }
        screenshot_path = browser.capture(question_id, artifacts)
        result = _evaluate_question(
            question=question,
            events=events,
            final=final,
            artifacts=artifacts,
            artifact_contents=artifact_contents,
            raw_path=raw_path,
            screenshot_path=screenshot_path,
            debug_trace=args.debug_trace,
        )
        results.append(result)
        transcript.append((question.prompt, result.final_answer))

    if browser_warning:
        results.insert(0, _warning_result("browser_smoke", browser_warning, raw_events_dir))

    browser.close()

    report = _build_report(
        suite=args.suite,
        suite_names=suite_names,
        dataset_paths=[str(path) for path in dataset_paths],
        session_id=session_id,
        base_url=base_url,
        frontend_url=frontend_url,
        health_config=health_config,
        results=results,
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    (out_dir / "chat_transcript.md").write_text(_chat_transcript(transcript), encoding="utf-8")

    print(
        json.dumps(
            {
                "passed": report["summary"]["failed"] == 0,
                "summary": report["summary"],
                "report": str(out_dir / "report.json"),
                "markdown": str(out_dir / "report.md"),
            },
            indent=2,
        )
    )
    return 0 if report["summary"]["failed"] == 0 else 1


def _expand_suite_alias(suite: str) -> list[str]:
    if suite == "generated_all":
        return [
            "generated_nested_customer_events",
            "generated_mixed_dataframe_numpy_bundle",
            "generated_custom_sensor_fleet",
            "generated_mixed_top_level_collection",
            "generated_multi_dataset",
        ]
    return [suite]


def _questions_for_suites(suite_names: list[str]) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for suite_name in suite_names:
        try:
            questions.extend(SUITES[suite_name])
        except KeyError as exc:
            raise SystemExit(f"Unknown suite: {suite_name}") from exc
    return questions


def _quick_subset(questions: list[EvalQuestion], suite: str) -> list[EvalQuestion]:
    if suite == "softbank_core":
        keep = {
            "inspect_file",
            "tabular_preview",
            "schema_summary",
            "top_countries",
            "country_bar_chart",
            "ambiguous_clean",
        }
        return [question for question in questions if question.id in keep]
    return questions[: min(3, len(questions))]


def _resolve_dataset_paths(args: argparse.Namespace, suite_names: list[str]) -> list[Path]:
    if suite_names == ["softbank_core"]:
        if not args.dataset:
            raise SystemExit("Missing --dataset or E2E_SOFTBANK_PKL_PATH for softbank_core.")
        path = Path(args.dataset).expanduser()
        if not path.exists():
            raise SystemExit(f"Dataset does not exist: {path}")
        return [path]

    dataset_dir = Path(args.dataset_dir or "agent_test_datasets").expanduser()
    _ensure_generated_datasets(dataset_dir)
    paths: list[Path] = []
    required = _required_generated_keys(suite_names)
    for key in required:
        env_name = f"AGENT_TEST_{key.upper()}_PKL"
        path = Path(os.environ.get(env_name, dataset_dir / GENERATED_DATASET_FILENAMES[key])).expanduser()
        if not path.exists():
            raise SystemExit(f"Generated dataset missing for {key}: {path}")
        paths.append(path)
    return paths


def _required_generated_keys(suite_names: list[str]) -> list[str]:
    keys: list[str] = []
    mapping = {
        "generated_nested_customer_events": ["nested_customer_events"],
        "generated_mixed_dataframe_numpy_bundle": ["mixed_dataframe_numpy_bundle"],
        "generated_custom_sensor_fleet": ["custom_sensor_fleet"],
        "generated_mixed_top_level_collection": ["mixed_top_level_collection"],
        "generated_multi_dataset": [
            "nested_customer_events",
            "mixed_dataframe_numpy_bundle",
            "custom_sensor_fleet",
        ],
    }
    for suite_name in suite_names:
        for key in mapping.get(suite_name, []):
            if key not in keys:
                keys.append(key)
    return keys


def _ensure_generated_datasets(dataset_dir: Path) -> None:
    missing = [name for name in GENERATED_DATASET_FILENAMES.values() if not (dataset_dir / name).exists()]
    if not missing:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.scripts.create_agent_test_datasets import write_agent_test_datasets

    write_agent_test_datasets(dataset_dir)


def _upload_required_datasets(
    base_url: str,
    session_id: str,
    dataset_paths: list[Path],
    *,
    existing_session: bool,
) -> list[DatasetUpload]:
    if existing_session:
        datasets = _json_get(base_url, f"/api/sessions/{session_id}/datasets").get("datasets", [])
        if datasets:
            return [
                DatasetUpload(
                    path=Path(dataset.get("original_filename", "restored.pkl")),
                    dataset_id=str(dataset["id"]),
                    filename=str(dataset.get("original_filename") or "restored.pkl"),
                    key=dataset.get("dataset_key"),
                )
                for dataset in datasets
            ]
    uploads: list[DatasetUpload] = []
    for path in dataset_paths:
        upload = _upload_pickle(base_url, session_id, path)
        dataset = upload["datasets"][0]
        uploads.append(
            DatasetUpload(
                path=path,
                dataset_id=str(dataset["id"]),
                filename=path.name,
                key=dataset.get("dataset_key"),
            )
        )
    return uploads


def _get_health_config(base_url: str) -> dict[str, Any]:
    try:
        return _json_get(base_url, "/health/config")
    except Exception:
        return {"agent_mode": "unknown", "verifier_mode": "unknown"}


def _create_session(base_url: str, name: str) -> dict[str, Any]:
    return _json_request(base_url, "POST", "/api/sessions", {"name": name})


def _active_dataset_id(base_url: str, session_id: str) -> str:
    session = _json_get(base_url, f"/api/sessions/{session_id}")
    dataset_id = session.get("active_dataset_id")
    if not dataset_id:
        datasets = _json_get(base_url, f"/api/sessions/{session_id}/datasets").get("datasets", [])
        if datasets:
            dataset_id = datasets[0]["id"]
    if not dataset_id:
        raise RuntimeError("No active dataset is available for this session.")
    return str(dataset_id)


def _activate_dataset(base_url: str, session_id: str, dataset_id: str) -> None:
    _json_request(base_url, "POST", f"/api/sessions/{session_id}/datasets/{dataset_id}/activate", {})


def _json_get(base_url: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_request(base_url: str, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _upload_pickle(base_url: str, session_id: str, dataset_path: Path) -> dict[str, Any]:
    boundary = f"----codex-eval-{int(time.time() * 1000)}"
    payload = bytearray()
    payload.extend(f"--{boundary}\r\n".encode())
    payload.extend(
        (
            f'Content-Disposition: form-data; name="files"; filename="{dataset_path.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
    )
    payload.extend(dataset_path.read_bytes())
    payload.extend(f"\r\n--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        f"{base_url}/api/sessions/{session_id}/datasets",
        data=bytes(payload),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _chat(
    base_url: str,
    session_id: str,
    dataset_id: str,
    message: str,
    *,
    approve_mutations: bool,
) -> list[dict[str, Any]]:
    events = _stream_chat(base_url, session_id, dataset_id, message)
    confirmation = _last_event(events, "confirmation_required")
    if confirmation and approve_mutations and confirmation.get("confirmation_id"):
        response = _json_request(
            base_url,
            "POST",
            f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/approve",
            {},
        )
        for event in response.get("events", []):
            if isinstance(event, dict):
                event["from_confirmation_approval"] = True
                events.append(event)
    return events


def _stream_chat(base_url: str, session_id: str, dataset_id: str, message: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url}/api/sessions/{session_id}/chat/stream",
        data=json.dumps({"message": message, "active_dataset_id": dataset_id}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    events: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            block: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\n")
                if line:
                    block.append(line)
                    continue
                events.extend(_parse_sse_block(block))
                block = []
            events.extend(_parse_sse_block(block))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        events.append({"type": "error", "message": body or str(exc)})
    except Exception as exc:
        events.append({"type": "error", "message": str(exc)})
    return events


def _parse_sse_block(lines: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[6:]))
        except json.JSONDecodeError as exc:
            events.append({"type": "error", "message": f"Unable to parse SSE JSON: {exc}"})
    return events


def _artifact_content(base_url: str, session_id: str, artifact_id: str) -> Any:
    request = urllib.request.Request(
        f"{base_url}/api/sessions/{session_id}/artifacts/{artifact_id}/content",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except Exception:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def _evaluate_question(
    *,
    question: EvalQuestion,
    events: list[dict[str, Any]],
    final: dict[str, Any],
    artifacts: list[dict[str, Any]],
    artifact_contents: dict[str, Any],
    raw_path: Path,
    screenshot_path: str | None,
    debug_trace: bool,
) -> QuestionResult:
    failure_reasons: list[str] = []
    warning_reasons: list[str] = []
    for check in ["no_internal_error", "no_stream_abort", "no_step_budget", "no_raw_json"]:
        ok, reason = _run_check(check, final, artifacts, artifact_contents, events)
        if not ok:
            failure_reasons.append(reason)
    for check in question.checks:
        ok, reason = _run_check(check, final, artifacts, artifact_contents, events)
        if not ok:
            failure_reasons.append(reason)
    for check in question.warnings:
        ok, reason = _run_check(check, final, artifacts, artifact_contents, events)
        if not ok:
            warning_reasons.append(reason)
    if debug_trace:
        warning_reasons.extend(_debug_warnings(events))

    final_answer = str(final.get("answer") or "")
    status = "fail" if failure_reasons else "warning" if warning_reasons else "pass"
    return QuestionResult(
        id=question.id,
        question=question.prompt,
        status=status,
        failure_reasons=failure_reasons,
        warning_reasons=warning_reasons,
        final_answer=final_answer,
        artifacts=_summarize_artifacts(artifacts, artifact_contents),
        execute_python_calls=sum(1 for event in events if event.get("type") == "code_started"),
        code_failed_count=sum(
            1 for event in events if event.get("type") == "code_result_summary" and event.get("ok") is False
        ),
        verifier_retries=sum(
            1
            for event in events
            if event.get("type") == "verifier_result" and str(event.get("severity")) == "retry"
        ),
        llm_verifier_called=_llm_verifier_called(events),
        llm_verifier_skipped=_llm_verifier_skipped(events),
        llm_verifier_fallback_used=_llm_verifier_fallback_used(events),
        llm_verifier_skip_reason=_llm_verifier_skip_reason(events) if debug_trace else None,
        state_changed=any(event.get("type") == "final_answer" and event.get("state_changed") is True for event in events),
        confirmation_required=any(event.get("type") == "confirmation_required" for event in events),
        raw_events_path=str(raw_path),
        screenshot_path=screenshot_path,
    )


def _run_check(
    check: str | dict[str, Any],
    final: dict[str, Any],
    artifacts: list[dict[str, Any]],
    artifact_contents: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[bool, str]:
    answer = str(final.get("answer") or "")
    answer_lower = answer.lower()
    event_blob = json.dumps(events, ensure_ascii=False, default=str)
    trace_blob = "\n".join(str(event.get("message") or "") for event in events if event.get("type") == "trace")

    if isinstance(check, dict):
        return _run_structured_check(check, answer, artifacts, artifact_contents, events)

    if check == "no_step_budget":
        return _ok("step budget" not in answer_lower and "internal step" not in answer_lower, "Answer hit step budget.")
    if check == "no_internal_error":
        bad = [text for text in GLOBAL_FORBIDDEN_STRINGS if text.lower() in event_blob.lower() or text.lower() in answer_lower]
        return _ok(not bad, f"Internal/error text leaked: {', '.join(bad[:3])}")
    if check == "no_stream_abort":
        return _ok("signal is aborted" not in event_blob.lower(), "Stream aborted.")
    if check == "no_raw_json":
        raw_markers = ("{'object_type':", "{'type':", "'attrs'", "... truncated ...")
        return _ok(not any(marker in answer for marker in raw_markers), "Final answer contains raw JSON/Python repr.")
    if check == "state_changed_false":
        return _ok(final.get("state_changed") is False, "State changed unexpectedly.")
    if check == "state_changed_true":
        return _ok(any(event.get("state_changed") is True for event in events), "State did not change.")
    if check == "table_artifact":
        return _ok(any(artifact.get("kind") == "table" for artifact in artifacts), "Missing table artifact.")
    if check == "chart_artifact":
        return _ok(any(artifact.get("kind") == "chart" for artifact in artifacts), "Missing chart artifact.")
    if check == "csv_artifact":
        return _ok(any(artifact.get("kind") == "csv" for artifact in artifacts), "Missing CSV artifact.")
    if check == "download_url_exists":
        return _ok(any(artifact.get("download_url") for artifact in artifacts if artifact.get("kind") == "csv"), "CSV download URL missing.")
    if check == "confirmation_required":
        return _ok(any(event.get("type") == "confirmation_required" for event in events), "Missing confirmation_required event.")
    if check == "state_not_changed_before_approval":
        before = [event for event in events if not event.get("from_confirmation_approval")]
        return _ok(not any(event.get("state_changed") is True for event in before), "State changed before confirmation approval.")
    if check == "no_python_execution":
        return _ok(not any(event.get("type") == "code_started" for event in events), "Python execution happened.")
    if check == "no_code_failures":
        return _ok(
            not any(event.get("type") == "code_result_summary" and event.get("ok") is False for event in events),
            "At least one code execution failed.",
        )
    if check == "no_name_error":
        return _ok("NameError" not in event_blob, "NameError found.")
    if check == "no_helper_import_error":
        return _ok("No module named" not in event_blob, "Helper import/module error found.")
    if check == "no_save_chart_contract_error":
        return _ok("Chart spec must include data" not in event_blob, "save_chart contract error found.")
    if check == "mentions_patent_metadata":
        return _ok(any(term in answer_lower for term in ("patent", "portfolio", "metadata")), "Answer does not mention patent/portfolio/metadata.")
    if check == "mentions_24410":
        normalized = answer.replace(",", "")
        return _ok("24410" in normalized or "24,410" in answer, "Answer does not mention the expected 24,410-ish row count.")
    if check == "mentions_record_meaning":
        return _ok("each record" in answer_lower or "record represents" in answer_lower, "Answer does not explain what a record represents.")
    if check == "mentions_representative_fields":
        fields = ("country", "doc_number", "title", "owners", "assignees", "filing_date", "status")
        return _ok(sum(field in answer_lower for field in fields) >= 4, "Answer lacks representative domain fields.")
    if check == "schema_terms":
        return _ok(all(term in answer_lower for term in ("scalar", "date", "list")), "Schema answer lacks scalar/date/list groups.")
    if check == "schema_domain_fields":
        return _ok(("owners" in answer_lower or "assignees" in answer_lower) and "filing" in answer_lower, "Schema answer lacks real domain fields.")
    if check == "no_pydantic_in_answer":
        return _ok("__pydantic" not in answer_lower and "__dict__" not in answer_lower, "Wrapper fields leaked in final answer.")
    if check == "no_wrapper_columns":
        return _ok(not _has_wrapper_columns(artifacts), "Wrapper columns leaked in artifact.")
    if check == "list_fields_structured":
        return _ok(not _table_has_python_repr_list(artifacts, artifact_contents), "List fields are Python repr strings in table artifact.")
    if check == "preview_table_domain_columns":
        required = {"country", "doc_number", "title", "owners", "assignees", "filing_date", "status"}
        return _ok(_any_table_has_columns(artifacts, artifact_contents, required, min_matches=5), "Preview table lacks required patent columns.")
    if check == "table_not_top_countries":
        return _ok(not _first_table_is_top_countries(artifacts, artifact_contents), "Preview/date request got a top-country table.")
    if check == "country_count_table":
        return _ok(_any_table_has_columns(artifacts, artifact_contents, {"country"}, 1) and _any_count_column(artifacts, artifact_contents), "Country count table missing country/count columns.")
    if check == "filings_by_year_table":
        return _ok(_has_filings_by_year_table(artifacts, artifact_contents), "Missing filings-by-year table.")
    if check == "filing_date_range_in_answer":
        return _ok("filing" in answer_lower and bool(re.search(r"20\d{2}|19\d{2}", answer)), "Filing date range not visible in answer.")
    if check == "no_duplicate_filings_tables":
        return _ok(len(_filings_by_year_tables(artifacts, artifact_contents)) <= 1, "Duplicate filings-by-year tables created.")
    if check == "no_duplicate_artifact_titles":
        titles = [str(artifact.get("title") or artifact.get("name") or "").lower() for artifact in artifacts]
        return _ok(len(titles) == len(set(titles)), "Duplicate artifact titles.")
    if check == "not_cn19_ca1_only":
        return _ok(not _looks_like_sample_only_country_result(artifacts, artifact_contents), "Country result looks sample-only CN=19/CA=1.")
    if check == "full_dataset_country_counts":
        return _ok(_full_count_evidence(artifacts, artifact_contents, {24410}), "No full-dataset country count evidence.")
    if check == "chart_data_valid":
        return _ok(_has_valid_chart_data(artifacts, artifact_contents), "Chart spec data/x/y invalid.")
    if check.startswith("chart_type_"):
        expected = check.removeprefix("chart_type_")
        spec = _first_chart_spec(artifacts, artifact_contents)
        return _ok(str(spec.get("chart_type") or "").lower() == expected, f"Expected chart_type {expected}.")
    if check == "country_chart_fields":
        spec = _first_chart_spec(artifacts, artifact_contents)
        return _ok(str(spec.get("x") or "").lower() == "country" and str(spec.get("y") or "").lower() in {"record_count", "count", "patent_count"}, "Country chart x/y fields invalid.")
    if check == "filings_chart_fields":
        spec = _first_chart_spec(artifacts, artifact_contents)
        return _ok(str(spec.get("x") or "").lower() in {"filing_year", "year"} and str(spec.get("y") or "").lower() in {"filing_count", "count"}, "Filings chart x/y fields invalid.")
    if check == "chart_title_not_fake":
        return _ok("fake agent chart" not in json.dumps(artifacts, ensure_ascii=False).lower(), "Fake chart title leaked.")
    if check == "clarification_answer":
        return _ok("please choose" in answer_lower or "clarification" in answer_lower or "which rule" in answer_lower, "Expected clarification answer.")
    if check == "history_answer":
        return _ok("branch" in answer_lower or "version" in answer_lower or "original upload" in answer_lower, "Mutation history answer lacks branch/version info.")
    if check == "delete_last_500_confirmation":
        confirmation = _last_event(events, "confirmation_required")
        text = json.dumps(confirmation, ensure_ascii=False).lower()
        return _ok("last 500" in text and ("affected_count" in text or "new_row_count" in text), "Delete-last-500 confirmation payload incomplete.")
    if check == "confirmation_payload_summary":
        confirmation = _last_event(events, "confirmation_required")
        required = ("operation_summary", "proposed_code", "state_impact", "rollback_note")
        return _ok(all(confirmation.get(key) for key in required), "Confirmation payload lacks human-readable summary fields.")
    if check == "full_scan_delete_empty_title":
        return _ok("full" in answer_lower and "scan" in answer_lower or bool(_last_event(events, "confirmation_required")), "Empty-title deletion did not show full scan or confirmation.")
    if check == "no_llm_skip_trace":
        return _ok("Skipping LLM verifier" not in trace_blob, "Public trace spammed LLM skip reason.")
    if check == "llm_verifier_called":
        return _ok(_llm_verifier_called(events), "LLM/semantic verifier was not called.")
    if check == "answer_not_too_short":
        return _ok(len(answer.split()) >= 35, "Conceptual answer is too short.")
    if check == "persistence_messages_visible":
        return _ok(True, "Persistence check placeholder.")

    return False, f"Unknown check: {check}"


def _run_structured_check(
    check: dict[str, Any],
    answer: str,
    artifacts: list[dict[str, Any]],
    artifact_contents: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[bool, str]:
    if "must_contain" in check:
        terms = [str(term).lower() for term in check["must_contain"]]
        return _ok(all(term in answer.lower() for term in terms), f"Answer missing required terms: {terms}")
    if "must_not_contain" in check:
        terms = [str(term) for term in check["must_not_contain"]]
        blob = answer + json.dumps(events, ensure_ascii=False, default=str)
        found = [term for term in terms if term in blob]
        return _ok(not found, f"Forbidden terms found: {found}")
    if "any_of" in check:
        terms = [str(term).lower() for term in check["any_of"]]
        return _ok(any(term in answer.lower() for term in terms), f"Answer missing any of: {terms}")
    if "regex" in check:
        pattern = str(check["regex"])
        return _ok(bool(re.search(pattern, answer, re.IGNORECASE)), f"Answer failed regex: {pattern}")
    if "max_execute_python_calls" in check:
        count = sum(1 for event in events if event.get("type") == "code_started")
        limit = int(check["max_execute_python_calls"])
        return _ok(count <= limit, f"Too many execute_python calls: {count} > {limit}")
    if "max_verifier_retries" in check:
        count = sum(1 for event in events if event.get("type") == "verifier_result" and event.get("severity") == "retry")
        limit = int(check["max_verifier_retries"])
        return _ok(count <= limit, f"Too many verifier retries: {count} > {limit}")
    if "table_required_columns" in check:
        required = {str(item).lower() for item in check["table_required_columns"]}
        return _ok(_any_table_has_columns(artifacts, artifact_contents, required, len(required)), f"Missing table columns: {sorted(required)}")
    if "table_forbidden_columns" in check:
        forbidden = {str(item).lower() for item in check["table_forbidden_columns"]}
        keys = set().union(*[_table_keys(doc) for doc in _table_documents(artifacts, artifact_contents)] or [set()])
        found = keys & forbidden
        return _ok(not found, f"Forbidden table columns found: {sorted(found)}")
    if "min_rows" in check:
        rows = max((_table_row_count(doc) for doc in _table_documents(artifacts, artifact_contents)), default=0)
        return _ok(rows >= int(check["min_rows"]), f"Table row count {rows} below minimum {check['min_rows']}.")
    if "min_columns" in check:
        cols = max((len(_table_keys(doc)) for doc in _table_documents(artifacts, artifact_contents)), default=0)
        return _ok(cols >= int(check["min_columns"]), f"Table column count {cols} below minimum {check['min_columns']}.")
    if "chart_type" in check:
        spec = _first_chart_spec(artifacts, artifact_contents)
        expected = str(check["chart_type"]).lower()
        return _ok(str(spec.get("chart_type") or "").lower() == expected, f"Expected chart type {expected}.")
    if "chart_x" in check or "chart_y" in check:
        spec = _first_chart_spec(artifacts, artifact_contents)
        x_ok = str(spec.get("x") or "").lower() in {str(item).lower() for item in check.get("chart_x", [spec.get("x")])}
        y_ok = str(spec.get("y") or "").lower() in {str(item).lower() for item in check.get("chart_y", [spec.get("y")])}
        return _ok(x_ok and y_ok, f"Chart x/y mismatch: x={spec.get('x')} y={spec.get('y')}")
    if "chart_data_min" in check:
        spec = _first_chart_spec(artifacts, artifact_contents)
        data = spec.get("data")
        count = len(data) if isinstance(data, list) else 0
        return _ok(count >= int(check["chart_data_min"]), f"Chart data length {count} below minimum.")
    return False, f"Unknown structured check: {check}"


def _ok(condition: bool, reason: str) -> tuple[bool, str]:
    return condition, "" if condition else reason


def _run_special_check(
    *,
    base_url: str,
    session_id: str,
    question: EvalQuestion,
    question_id: str,
    raw_events_dir: Path,
    browser: BrowserSmoke,
    debug_trace: bool,
    restart_backend_check: bool,
) -> QuestionResult:
    del debug_trace
    events: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []
    if question.special == "refresh_persistence":
        messages = _json_get(base_url, f"/api/sessions/{session_id}/messages").get("messages", [])
        artifacts = _json_get(base_url, f"/api/sessions/{session_id}/artifacts")
        events = [{"type": "persistence_check", "message_count": len(messages), "artifact_count": len(artifacts)}]
        if not messages:
            failures.append("No messages were restored from the session endpoint.")
        if browser.enabled and not browser.available:
            warnings.append("Browser smoke unavailable, API persistence checked only.")
        screenshot_path = browser.capture(question_id, artifacts if isinstance(artifacts, list) else [])
    elif question.special == "restart_persistence":
        screenshot_path = None
        if not restart_backend_check:
            warnings.append("Backend restart check skipped; pass --restart-backend-check to run it manually.")
        else:
            warnings.append("Automated backend restart is environment-specific; API persistence endpoints remain checked.")
    else:
        screenshot_path = None
        warnings.append(f"Unknown special check: {question.special}")

    raw_path = raw_events_dir / f"{question_id}.json"
    raw_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "fail" if failures else "warning" if warnings else "pass"
    return QuestionResult(
        id=question.id,
        question=question.prompt,
        status=status,
        failure_reasons=failures,
        warning_reasons=warnings,
        final_answer="Scripted persistence check.",
        artifacts=[],
        execute_python_calls=0,
        code_failed_count=0,
        verifier_retries=0,
        llm_verifier_called=False,
        llm_verifier_skipped=False,
        llm_verifier_fallback_used=False,
        llm_verifier_skip_reason=None,
        state_changed=False,
        confirmation_required=False,
        raw_events_path=str(raw_path),
        screenshot_path=screenshot_path,
    )


class BrowserSmoke:
    def __init__(self, *, enabled: bool, frontend_url: str, session_id: str, screenshots_dir: Path) -> None:
        self.enabled = enabled
        self.frontend_url = frontend_url
        self.session_id = session_id
        self.screenshots_dir = screenshots_dir
        self.available = False
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    def start(self) -> str | None:
        if not self.enabled:
            return None
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            return "Playwright is not installed; browser smoke and screenshots were skipped."
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
            self._page = self._browser.new_page(viewport={"width": 1440, "height": 1000})
            self.available = True
            return None
        except Exception as exc:
            return f"Unable to start Playwright browser smoke: {exc}"

    def capture(self, question_id: str, artifacts: list[dict[str, Any]]) -> str | None:
        if not self.enabled or not self.available or self._page is None:
            return None
        try:
            url = f"{self.frontend_url}/?session={urllib.parse.quote(self.session_id)}"
            self._page.goto(url, wait_until="networkidle", timeout=20_000)
            self._page.wait_for_timeout(1000)
            path = self.screenshots_dir / f"{question_id}.png"
            self._page.screenshot(path=str(path), full_page=False)
            self._smoke_assert_artifacts(artifacts)
            return str(path)
        except Exception:
            return None

    def _smoke_assert_artifacts(self, artifacts: list[dict[str, Any]]) -> None:
        if self._page is None:
            return
        if any(artifact.get("kind") == "chart" for artifact in artifacts):
            svg_count = self._page.locator("svg").count()
            mark_count = self._page.locator("svg rect, svg path, svg circle, svg polyline").count()
            if svg_count < 1 or mark_count < 1:
                raise RuntimeError("Chart smoke failed: no visible SVG marks.")
        if any(artifact.get("kind") == "table" for artifact in artifacts):
            if self._page.locator("table").count() < 1:
                raise RuntimeError("Table smoke failed: no table element.")
        if any(artifact.get("kind") == "csv" for artifact in artifacts):
            if self._page.get_by_text("Download", exact=False).count() < 1:
                raise RuntimeError("CSV smoke failed: no download affordance.")

    def close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass


def _warning_result(identifier: str, warning: str, raw_events_dir: Path) -> QuestionResult:
    raw_path = raw_events_dir / f"{identifier}.json"
    raw_path.write_text(json.dumps([{"type": "warning", "message": warning}], indent=2), encoding="utf-8")
    return QuestionResult(
        id=identifier,
        question="Browser smoke setup",
        status="warning",
        failure_reasons=[],
        warning_reasons=[warning],
        final_answer=warning,
        artifacts=[],
        execute_python_calls=0,
        code_failed_count=0,
        verifier_retries=0,
        llm_verifier_called=False,
        llm_verifier_skipped=False,
        llm_verifier_fallback_used=False,
        llm_verifier_skip_reason=None,
        state_changed=False,
        confirmation_required=False,
        raw_events_path=str(raw_path),
    )


def _debug_warnings(events: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for event in events:
        if event.get("type") == "trace" and "Verifier completed using deterministic checks" in str(event.get("message")):
            warnings.append("LLM verifier fallback trace observed.")
            break
    return warnings


def _last_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("type") == event_type:
            return event
    return {}


def _summarize_artifacts(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for artifact in artifacts:
        content = artifact_contents.get(str(artifact.get("id")))
        summary = {
            "id": artifact.get("id"),
            "type": artifact.get("type") or artifact.get("kind"),
            "kind": artifact.get("kind"),
            "title": artifact.get("title") or artifact.get("name"),
            "download_url": artifact.get("download_url"),
        }
        if artifact.get("kind") == "table" and isinstance(content, dict):
            summary.update(
                {
                    "row_count": content.get("row_count") or _table_row_count(content),
                    "column_count": content.get("total_column_count") or content.get("column_count") or len(_table_keys(content)),
                    "preview_row_count": content.get("preview_row_count"),
                    "columns": sorted(_table_keys(content))[:40],
                }
            )
        if artifact.get("kind") == "chart":
            spec = _artifact_chart_spec(artifact, content)
            data = spec.get("data")
            summary.update(
                {
                    "chart_type": spec.get("chart_type"),
                    "x": spec.get("x"),
                    "y": spec.get("y"),
                    "data_length": len(data) if isinstance(data, list) else 0,
                }
            )
        if artifact.get("kind") == "csv":
            summary["row_count"] = (artifact.get("metadata") or {}).get("row_count")
        summaries.append(summary)
    return summaries


def _build_report(
    *,
    suite: str,
    suite_names: list[str],
    dataset_paths: list[str],
    session_id: str,
    base_url: str,
    frontend_url: str,
    health_config: dict[str, Any],
    results: list[QuestionResult],
) -> dict[str, Any]:
    passed = sum(1 for result in results if result.status == "pass")
    warnings = sum(1 for result in results if result.status == "warning")
    failed = sum(1 for result in results if result.status == "fail")
    return {
        "summary": {
            "total": len(results),
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
        },
        "suite": suite,
        "suite_names": suite_names,
        "dataset_paths": dataset_paths,
        "session_id": session_id,
        "agent_mode": health_config.get("agent_mode"),
        "verifier_mode": health_config.get("verifier_mode"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "frontend_url": frontend_url,
        "questions": [
            {
                "id": result.id,
                "question": result.question,
                "status": result.status,
                "failure_reasons": result.failure_reasons,
                "warning_reasons": result.warning_reasons,
                "final_answer": result.final_answer,
                "artifacts": result.artifacts,
                "execute_python_calls": result.execute_python_calls,
                "code_failed_count": result.code_failed_count,
                "verifier_retries": result.verifier_retries,
                "llm_verifier_called": result.llm_verifier_called,
                "llm_verifier_skipped": result.llm_verifier_skipped,
                "llm_verifier_fallback_used": result.llm_verifier_fallback_used,
                "llm_verifier_skip_reason": result.llm_verifier_skip_reason,
                "state_changed": result.state_changed,
                "confirmation_required": result.confirmation_required,
                "raw_events_path": result.raw_events_path,
                "screenshot_path": result.screenshot_path,
            }
            for result in results
        ],
    }


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Data Analysis Agent Evaluation Report",
        "",
        f"- Dataset path(s): {', '.join(f'`{path}`' for path in report['dataset_paths'])}",
        f"- Suite: `{report['suite']}` ({', '.join(report['suite_names'])})",
        f"- Session id: `{report['session_id']}`",
        f"- Agent mode: `{report.get('agent_mode')}`",
        f"- Verifier mode: `{report.get('verifier_mode')}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Backend URL: `{report['base_url']}`",
        f"- Frontend URL: `{report['frontend_url']}`",
        f"- Total questions: {summary['total']}",
        f"- Pass/fail/warning: {summary['passed']} pass, {summary['warnings']} warning, {summary['failed']} fail",
        "",
        "## Summary",
        "",
        "| ID | Question | Status | Artifacts | Exec calls | Retries | LLM verifier | State changed | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in report["questions"]:
        artifact_labels = ", ".join(str(artifact.get("kind") or artifact.get("type")) for artifact in item["artifacts"]) or "-"
        notes = "; ".join(item["failure_reasons"][:2] or item["warning_reasons"][:2])
        lines.append(
            "| {id} | {question} | {status} | {artifacts} | {exec_calls} | {retries} | {llm} | {state} | {notes} |".format(
                id=item["id"],
                question=_md_cell(item["question"][:90]),
                status=item["status"],
                artifacts=_md_cell(artifact_labels),
                exec_calls=item["execute_python_calls"],
                retries=item["verifier_retries"],
                llm=_llm_report_cell(item),
                state="yes" if item["state_changed"] else "no",
                notes=_md_cell(notes),
            )
        )
    for item in report["questions"]:
        lines.extend(
            [
                "",
                f"## {item['id']}: {item['status'].upper()}",
                "",
                f"**Question:** {item['question']}",
                "",
                "**Final answer excerpt:**",
                "",
                _excerpt(item["final_answer"], 900) or "_No final answer._",
                "",
            ]
        )
        if item["failure_reasons"]:
            lines.extend(["**Failure reasons:**", *[f"- {reason}" for reason in item["failure_reasons"]], ""])
        if item["warning_reasons"]:
            lines.extend(["**Warning reasons:**", *[f"- {reason}" for reason in item["warning_reasons"]], ""])
        lines.extend(
            [
                "**Artifacts created:**",
                *_artifact_markdown_lines(item["artifacts"]),
                "",
                "**Trace summary:**",
                f"- execute_python calls: {item['execute_python_calls']}",
                f"- code failures: {item['code_failed_count']}",
                f"- verifier retries: {item['verifier_retries']}",
                f"- LLM verifier called: {item['llm_verifier_called']}",
                f"- LLM verifier skipped: {item.get('llm_verifier_skipped', False)}",
                f"- LLM verifier fallback used: {item.get('llm_verifier_fallback_used', False)}",
                f"- confirmation_required: {item['confirmation_required']}",
                f"- state_changed: {item['state_changed']}",
                "",
                f"Raw events: `{item['raw_events_path']}`",
            ]
        )
        if item.get("screenshot_path"):
            lines.append(f"Screenshot: `{item['screenshot_path']}`")
        lines.extend(["", "<details>", "<summary>Generated Python code snippets</summary>", ""])
        raw_path = Path(item["raw_events_path"])
        code_snippets = _code_snippets_from_raw(raw_path)
        if code_snippets:
            for index, code in enumerate(code_snippets, start=1):
                lines.extend([f"### Code {index}", "", "```python", code[:5000], "```", ""])
        else:
            lines.append("_No Python code snippets._")
        lines.extend(["</details>", "", "Reviewer notes:", ""])
    return "\n".join(lines)


def _chat_transcript(transcript: list[tuple[str, str]]) -> str:
    lines = ["# Chat Transcript", ""]
    for index, (question, answer) in enumerate(transcript, start=1):
        lines.extend([f"## Q{index}", "", "**User**", "", question, "", "**Assistant**", "", answer or "_No answer._", ""])
    return "\n".join(lines)


def _artifact_markdown_lines(artifacts: list[dict[str, Any]]) -> list[str]:
    if not artifacts:
        return ["- None"]
    lines: list[str] = []
    for artifact in artifacts:
        if artifact.get("kind") == "chart":
            detail = f"{artifact.get('chart_type')} x={artifact.get('x')} y={artifact.get('y')} data={artifact.get('data_length')}"
        elif artifact.get("kind") == "table":
            detail = f"rows={artifact.get('row_count')} columns={artifact.get('column_count')}"
        elif artifact.get("kind") == "csv":
            detail = f"download={artifact.get('download_url')}"
        else:
            detail = ""
        lines.append(f"- {artifact.get('kind')}: {artifact.get('title')} `{artifact.get('id')}` {detail}")
    return lines


def _code_snippets_from_raw(path: Path) -> list[str]:
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(event.get("code")) for event in events if event.get("type") == "code_started" and event.get("code")]


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _llm_report_cell(item: dict[str, Any]) -> str:
    if item.get("llm_verifier_called"):
        return "called"
    if item.get("llm_verifier_fallback_used"):
        return "fallback"
    if item.get("llm_verifier_skipped"):
        return "skipped"
    return "no"


def _excerpt(text: str, limit: int) -> str:
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug[:60] or "question"


def _llm_verifier_called(events: list[dict[str, Any]]) -> bool:
    for event in events:
        message = str(event.get("message") or "")
        if "Semantic verifier reviewed" in message:
            return True
        if event.get("type") == "verifier_result" and str(event.get("source")) in {"llm", "hybrid"}:
            return True
    return False


def _llm_verifier_fallback_used(events: list[dict[str, Any]]) -> bool:
    for event in events:
        message = str(event.get("message") or "")
        if "deterministic checks" in message.lower():
            return True
        if event.get("type") == "verifier_result" and str(event.get("source")) == "deterministic":
            metadata = event.get("metadata")
            if isinstance(metadata, dict) and metadata.get("llm_fallback"):
                return True
    return False


def _llm_verifier_skipped(events: list[dict[str, Any]]) -> bool:
    if _llm_verifier_called(events):
        return False
    return any(event.get("type") == "verifier_result" for event in events)


def _llm_verifier_skip_reason(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") != "trace":
            continue
        message = str(event.get("message") or "")
        if "Skipping LLM verifier" in message or "LLM verifier" in message:
            return message
    return None


def _has_wrapper_columns(artifacts: list[dict[str, Any]]) -> bool:
    for artifact in artifacts:
        columns = artifact.get("columns") or artifact.get("metadata", {}).get("columns") or []
        for column in columns:
            key = column.get("key") if isinstance(column, dict) else column
            if str(key) in FORBIDDEN_COLUMNS:
                return True
    return False


def _table_documents(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("kind") != "table":
            continue
        content = artifact_contents.get(str(artifact.get("id")))
        if isinstance(content, dict):
            documents.append(content)
    return documents


def _table_keys(document: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    columns = document.get("columns") or document.get("display_columns") or []
    if isinstance(columns, list):
        for column in columns:
            key = column.get("key") if isinstance(column, dict) else column
            if key is not None:
                keys.add(str(key).lower())
    rows = document.get("rows")
    if isinstance(rows, list):
        for row in rows[:3]:
            if isinstance(row, dict):
                keys.update(str(key).lower() for key in row)
    return keys


def _table_row_count(document: dict[str, Any]) -> int:
    for key in ("row_count", "preview_row_count"):
        value = document.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    rows = document.get("rows")
    return len(rows) if isinstance(rows, list) else 0


def _any_table_has_columns(
    artifacts: list[dict[str, Any]],
    artifact_contents: dict[str, Any],
    required: set[str],
    min_matches: int,
) -> bool:
    for document in _table_documents(artifacts, artifact_contents):
        keys = _table_keys(document)
        if len(keys & {key.lower() for key in required}) >= min_matches:
            return True
    return False


def _any_count_column(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> bool:
    count_keys = {"count", "record_count", "patent_count", "filing_count", "total_revenue", "revenue"}
    return any(_table_keys(document) & count_keys for document in _table_documents(artifacts, artifact_contents))


def _table_has_python_repr_list(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> bool:
    for document in _table_documents(artifacts, artifact_contents):
        rows = document.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows[:20]:
            if not isinstance(row, dict):
                continue
            for value in row.values():
                if isinstance(value, str) and re.match(r"^\['.*'\]$", value):
                    return True
    return False


def _first_table_is_top_countries(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> bool:
    documents = _table_documents(artifacts, artifact_contents)
    if not documents:
        return False
    first = documents[0]
    title = str(first.get("title") or "").lower()
    keys = _table_keys(first)
    return "top countr" in title or (keys <= {"country", "count", "record_count", "patent_count"} and "country" in keys)


def _has_filings_by_year_table(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> bool:
    return bool(_filings_by_year_tables(artifacts, artifact_contents))


def _filings_by_year_tables(
    artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for document in _table_documents(artifacts, artifact_contents):
        keys = _table_keys(document)
        if keys & {"filing_year", "year"} and keys & {"filing_count", "count", "record_count"}:
            output.append(document)
    return output


def _full_count_evidence(
    artifacts: list[dict[str, Any]],
    artifact_contents: dict[str, Any],
    expected_counts: set[int],
) -> bool:
    for document in _table_documents(artifacts, artifact_contents):
        for key in ("analyzed_row_count", "source_row_count", "source_total_row_count"):
            value = document.get(key)
            if isinstance(value, (int, float)) and int(value) in expected_counts:
                return True
        rows = document.get("rows")
        if isinstance(rows, list):
            total = 0
            found = False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for count_key in ("count", "record_count", "patent_count", "filing_count"):
                    value = row.get(count_key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        total += int(value)
                        found = True
                        break
            if found and total in expected_counts:
                return True
    return False


def _looks_like_sample_only_country_result(
    artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]
) -> bool:
    for document in _table_documents(artifacts, artifact_contents):
        rows = document.get("rows")
        if not isinstance(rows, list):
            continue
        country_counts: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            country = row.get("country")
            count = row.get("record_count") or row.get("count")
            if country is not None and isinstance(count, (int, float)) and not isinstance(count, bool):
                country_counts[str(country)] = int(count)
        if country_counts == {"CN": 19, "CA": 1}:
            return True
        if country_counts and sum(country_counts.values()) == 20:
            return True
    return False


def _has_valid_chart_data(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> bool:
    for artifact in artifacts:
        if artifact.get("kind") != "chart":
            continue
        spec = _artifact_chart_spec(artifact, artifact_contents.get(str(artifact.get("id"))))
        data = spec.get("data")
        x = spec.get("x")
        y = spec.get("y")
        if isinstance(data, list) and data and isinstance(data[0], dict) and x in data[0] and y in data[0]:
            return True
    return False


def _first_chart_spec(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> dict[str, Any]:
    for artifact in artifacts:
        if artifact.get("kind") == "chart":
            return _artifact_chart_spec(artifact, artifact_contents.get(str(artifact.get("id"))))
    return {}


def _artifact_chart_spec(artifact: dict[str, Any], content: Any) -> dict[str, Any]:
    spec = (
        artifact.get("chart_spec")
        or artifact.get("metadata", {}).get("chart_spec")
        or (content.get("chart_spec") if isinstance(content, dict) else None)
        or {}
    )
    return spec if isinstance(spec, dict) else {}


SUITES: dict[str, list[EvalQuestion]] = {
    "softbank_core": [
        EvalQuestion(
            "inspect_file",
            "What's in this file?",
            [
                "state_changed_false",
                {"max_execute_python_calls": 3},
                "mentions_patent_metadata",
                "mentions_24410",
                "no_raw_json",
                "mentions_representative_fields",
            ],
            ["answer_not_too_short"],
        ),
        EvalQuestion(
            "tabular_preview",
            "Convert this dataset into a tabular preview if possible. Show the inferred columns and the first 5 rows.",
            [
                "table_artifact",
                {"min_rows": 5},
                {
                    "table_required_columns": [
                        "country",
                        "doc_number",
                        "title",
                        "owners",
                        "assignees",
                        "filing_date",
                        "status",
                    ]
                },
                {"table_forbidden_columns": list(FORBIDDEN_COLUMNS)},
                "list_fields_structured",
                "state_changed_false",
            ],
        ),
        EvalQuestion(
            "schema_summary",
            "Summarize the schema of this patent metadata dataset. Which fields are scalar fields, which are dates, and which are list-like fields?",
            ["schema_terms", "schema_domain_fields", "no_pydantic_in_answer", "state_changed_false"],
        ),
        EvalQuestion(
            "top_countries",
            "Show the top countries by number of patent records. Return both a table and a short explanation.",
            ["table_artifact", "country_count_table", "full_dataset_country_counts", "not_cn19_ca1_only", "state_changed_false"],
        ),
        EvalQuestion(
            "filing_date_range",
            "What is the filing date range of this portfolio? Also summarize the number of filings by year in a table.",
            [
                "filing_date_range_in_answer",
                "table_artifact",
                "filings_by_year_table",
                "no_duplicate_filings_tables",
                {"max_execute_python_calls": 3},
                "state_changed_false",
            ],
        ),
        EvalQuestion(
            "country_pie_chart",
            "create a pie chart for country",
            ["chart_artifact", "chart_type_pie", "country_chart_fields", "chart_data_valid", "no_name_error", "no_save_chart_contract_error"],
        ),
        EvalQuestion(
            "country_bar_chart",
            "Create a bar chart showing the number of patent records by country.",
            ["chart_artifact", "chart_type_bar", "country_chart_fields", "chart_data_valid"],
        ),
        EvalQuestion(
            "filings_line_chart",
            "Create a line chart showing filings by year based on filing_date.",
            ["chart_artifact", "chart_type_line", "filings_chart_fields", "chart_data_valid", "chart_title_not_fake"],
        ),
        EvalQuestion(
            "compound_table_chart",
            "Show the top 10 countries by record count as a table, then create a bar chart from the same data.",
            ["table_artifact", "chart_artifact", "country_count_table", "country_chart_fields", "state_changed_false"],
        ),
        EvalQuestion(
            "csv_export",
            "Export the top 10 countries table as CSV without changing the dataset.",
            ["csv_artifact", "download_url_exists", "state_changed_false"],
        ),
        EvalQuestion("ambiguous_clean", "Remove bad records.", ["clarification_answer", "no_python_execution", "state_changed_false"]),
        EvalQuestion("delete_last_500", "delete last 500 entries", ["confirmation_required", "delete_last_500_confirmation", "state_not_changed_before_approval"]),
        EvalQuestion("mutation_history", "Show me the mutation history so far.", ["history_answer", "no_name_error", "state_changed_false"]),
        EvalQuestion("persistence_refresh_check", "Scripted refresh persistence check", ["persistence_messages_visible"], special="refresh_persistence"),
        EvalQuestion("persistence_restart_check", "Scripted backend restart persistence check", [], special="restart_persistence"),
    ],
    "generated_nested_customer_events": [
        EvalQuestion("inspect_nested", "What's in this file? Explain the structure in plain English.", [{"any_of": ["customer", "event", "support", "order"]}, {"any_of": ["metadata", "customers", "lookup_tables"]}], dataset_key="nested_customer_events"),
        EvalQuestion("customer_preview", "Show a tabular preview of customers with customer_id, country, segment, joined_at, churn_risk.", ["table_artifact", {"table_required_columns": ["customer_id", "country", "segment", "joined_at", "churn_risk"]}], dataset_key="nested_customer_events"),
        EvalQuestion("normalize_events", "Normalize all purchase events into a table with customer_id, event_id, timestamp, channel, event_type, order_total.", ["table_artifact", {"table_required_columns": ["customer_id", "event_id", "timestamp", "channel", "event_type", "order_total"]}, {"min_rows": 1}], dataset_key="nested_customer_events"),
        EvalQuestion("revenue_by_country_chart", "Create a bar chart of total order revenue by country.", ["chart_artifact", "chart_type_bar", "chart_data_valid", {"chart_x": ["country"]}], dataset_key="nested_customer_events"),
        EvalQuestion("product_revenue", "Which products generate the most revenue? Remember items are nested inside events. Show a table.", ["table_artifact", {"any_of": ["product", "sku", "revenue"]}], dataset_key="nested_customer_events"),
        EvalQuestion("support_churn", "Find customers with high churn risk and support tickets. Show a table.", ["table_artifact", {"any_of": ["churn", "support", "ticket"]}], dataset_key="nested_customer_events"),
        EvalQuestion("export_risk_flags", "Export all risk flags as a CSV with customer_id, event_id, flag_type, severity.", ["csv_artifact", "state_changed_false"], dataset_key="nested_customer_events"),
        EvalQuestion("ambiguous_clean_nested", "Clean this dataset.", ["clarification_answer", "state_changed_false"], dataset_key="nested_customer_events"),
    ],
    "generated_mixed_dataframe_numpy_bundle": [
        EvalQuestion("inspect_bundle", "What's in this file? List all top-level keys and object types.", [{"must_contain": ["users", "orders"]}, {"any_of": ["embedding", "cohort", "array"]}], dataset_key="mixed_dataframe_numpy_bundle"),
        EvalQuestion("list_shapes", "List all tables and arrays with their shapes.", ["table_artifact", {"any_of": ["users", "orders", "daily_metrics", "embedding", "cohort"]}], dataset_key="mixed_dataframe_numpy_bundle"),
        EvalQuestion("join_revenue_country", "Join users and orders on user_id, then show revenue by country as a table and chart.", ["table_artifact", "chart_artifact", "state_changed_false"], dataset_key="mixed_dataframe_numpy_bundle"),
        EvalQuestion("daily_active_users_chart", "Create a line chart of daily active_users over time.", ["chart_artifact", "chart_type_line", "chart_data_valid"], dataset_key="mixed_dataframe_numpy_bundle"),
        EvalQuestion("latency_error_scatter", "Create a scatter chart of latency_ms_p95 versus error_rate.", ["chart_artifact", "chart_type_scatter", "chart_data_valid"], dataset_key="mixed_dataframe_numpy_bundle"),
        EvalQuestion("embedding_summary", "Inspect the user_embedding_matrix and summarize its shape and basic statistics.", [{"any_of": ["500", "16"]}, "state_changed_false"], dataset_key="mixed_dataframe_numpy_bundle"),
        EvalQuestion("order_status_category", "Compare paid versus refunded orders by product category.", ["table_artifact"], dataset_key="mixed_dataframe_numpy_bundle"),
        EvalQuestion("export_top_users", "Export the top 20 users by total gross_revenue as CSV.", ["csv_artifact", "state_changed_false"], dataset_key="mixed_dataframe_numpy_bundle"),
    ],
    "generated_custom_sensor_fleet": [
        EvalQuestion("inspect_custom", "What's in this file? Explain the custom class structure.", [{"any_of": ["SensorFleet", "sensor", "readings"]}], dataset_key="custom_sensor_fleet"),
        EvalQuestion("readings_table", "Convert sensor readings into a table with sensor_id, timestamp, site, zone, temperature_c, vibration_g, battery_pct.", ["table_artifact", {"table_required_columns": ["sensor_id", "timestamp", "site", "zone", "temperature_c", "vibration_g", "battery_pct"]}], dataset_key="custom_sensor_fleet"),
        EvalQuestion("average_temperature_line", "Create a line chart of average temperature by hour.", ["chart_artifact", "chart_type_line", "chart_data_valid"], dataset_key="custom_sensor_fleet"),
        EvalQuestion("alert_counts_bar", "Create a bar chart of alert counts by alert type.", ["chart_artifact", "chart_type_bar", "chart_data_valid"], dataset_key="custom_sensor_fleet"),
        EvalQuestion("high_vibration_table", "Find sensors with high vibration alerts and show a table.", ["table_artifact", {"any_of": ["sensor", "vibration", "alert"]}], dataset_key="custom_sensor_fleet"),
        EvalQuestion("export_alerts", "Export all alerts as CSV with sensor_id, timestamp, alert_type, severity, value.", ["csv_artifact", "state_changed_false"], dataset_key="custom_sensor_fleet"),
        EvalQuestion("remove_low_battery", "Remove readings with battery_pct below 5, but ask for confirmation first.", ["confirmation_required", "state_not_changed_before_approval"], dataset_key="custom_sensor_fleet"),
    ],
    "generated_mixed_top_level_collection": [
        EvalQuestion("inspect_mixed", "What's in this file? Explain each top-level element and its type.", [{"any_of": ["mixed", "top-level", "DataFrame", "numpy", "tuple"]}], dataset_key="mixed_top_level_collection"),
        EvalQuestion("preview_tabular_elements", "Try to convert each tabular-looking element into a preview table.", ["table_artifact"], dataset_key="mixed_top_level_collection"),
        EvalQuestion("nested_records_fields", "Find all nested records and show their common fields.", [{"any_of": ["record", "field", "payload"]}], dataset_key="mixed_top_level_collection"),
        EvalQuestion("element_type_summary", "Create a summary table of top-level element types.", ["table_artifact", {"table_required_columns": ["element_index", "type"]}], dataset_key="mixed_top_level_collection"),
    ],
    "generated_multi_dataset": [
        EvalQuestion("list_all_datasets", "List all datasets in this session. For each dataset, show object type, row count or length, and representative fields.", ["table_artifact", "state_changed_false"]),
        EvalQuestion("compare_datasets", "Compare the uploaded datasets by object type, approximate row count or length, schema style, and key fields.", ["table_artifact", {"any_of": ["nested", "NumPy", "custom", "class"]}]),
        EvalQuestion("identify_join_keys", "Inspect the uploaded datasets and identify possible join keys, if any. Do not join yet.", ["state_changed_false", {"any_of": ["join", "key", "no direct"]}]),
        EvalQuestion("cross_dataset_chart", "Create a comparison chart showing approximate record counts for each uploaded dataset.", ["table_artifact", "chart_artifact", "chart_type_bar", "chart_data_valid"]),
    ],
}


if __name__ == "__main__":
    raise SystemExit(main())

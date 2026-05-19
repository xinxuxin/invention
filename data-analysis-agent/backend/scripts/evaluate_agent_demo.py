from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


QUICK_QUESTIONS = [
    {
        "id": "inspect_file",
        "prompt": "What is in this file? Please inspect the object type, size, structure, and show a few representative examples.",
        "expects": [
            "no_step_budget",
            "mentions_patent_metadata",
            "state_changed_false",
            "no_raw_dict_preview",
            "no_public_verifier_exception",
        ],
    },
    {
        "id": "tabular_preview",
        "prompt": "Convert this dataset into a tabular preview if possible. Show the inferred columns and the first 5 rows.",
        "expects": ["table_artifact", "no_wrapper_columns", "state_changed_false"],
    },
    {
        "id": "schema",
        "prompt": "Summarize the schema of this patent metadata dataset. Which fields are scalar fields, which are dates, and which are list-like fields?",
        "expects": ["schema_terms", "no_step_budget"],
    },
    {
        "id": "top_countries",
        "prompt": "Show the top countries by number of patent records. Return both a table and a short explanation.",
        "expects": ["table_artifact", "state_changed_false"],
    },
    {
        "id": "filings_chart",
        "prompt": "Create a line chart or bar chart showing filings by year based on filing_date.",
        "expects": ["chart_artifact", "chart_data_valid"],
    },
    {
        "id": "export_top5",
        "prompt": "Export that top-5 table as CSV, but do not change the working dataset.",
        "expects": ["csv_artifact", "state_changed_false"],
    },
    {
        "id": "ambiguous_remove_bad",
        "prompt": "Remove bad records.",
        "expects": ["clarification_answer", "no_python_execution", "state_changed_false"],
    },
    {
        "id": "mutation_history",
        "prompt": "Show me the mutation history so far.",
        "expects": ["history_answer", "no_name_error", "state_changed_false"],
    },
    {
        "id": "confirmation_summary",
        "prompt": "Create a cleaned working dataset that removes records with missing titles. This should mutate the current dataset.",
        "expects": ["confirmation_required", "confirmation_payload_summary"],
    },
]

FORBIDDEN_COLUMNS = {
    "__dict__",
    "__pydantic_extra__",
    "__pydantic_fields_set__",
    "__pydantic_private__",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a quick Data Analysis Agent demo evaluation.")
    parser.add_argument("--quick", action="store_true", help="Run the built-in quick SoftBank-style rubric.")
    parser.add_argument("--dataset", default=os.environ.get("E2E_SOFTBANK_PKL_PATH"))
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--out-dir", default="eval_reports")
    args = parser.parse_args()

    if not args.dataset:
        print("Missing --dataset or E2E_SOFTBANK_PKL_PATH.", file=sys.stderr)
        return 2

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(f"Dataset does not exist: {dataset_path}", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    session = _json_request(base_url, "POST", "/api/sessions", {"name": f"Eval {int(time.time())}"})
    session_id = session["id"]
    upload = _upload_pickle(base_url, session_id, dataset_path)
    dataset_id = upload["datasets"][0]["id"]

    results = []
    for question in QUICK_QUESTIONS:
        events = _chat(base_url, session_id, dataset_id, question["prompt"])
        final = next((event for event in events if event.get("type") == "final_answer"), {})
        confirmation = next((event for event in events if event.get("type") == "confirmation_required"), {})
        artifacts = [event["artifact"] for event in events if event.get("type") == "artifact_created"]
        checks = _run_checks(question["expects"], final, artifacts, events, confirmation)
        results.append(
            {
                "id": question["id"],
                "prompt": question["prompt"],
                "passed": all(checks.values()),
                "checks": checks,
                "final_answer": final.get("answer"),
                "state_changed": final.get("state_changed"),
                "artifacts": [
                    {
                        "id": artifact.get("id"),
                        "kind": artifact.get("kind"),
                        "title": artifact.get("title") or artifact.get("name"),
                        "metadata": artifact.get("metadata"),
                    }
                    for artifact in artifacts
                ],
                "trace": [event.get("message") for event in events if event.get("type") == "trace"],
                "confirmation": confirmation,
                "event_types": [event.get("type") for event in events],
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "base_url": base_url,
        "dataset": str(dataset_path),
        "session_id": session_id,
        "passed": all(item["passed"] for item in results),
        "results": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(out_dir / "report.json")}, indent=2))
    return 0 if report["passed"] else 1


def _json_request(base_url: str, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _upload_pickle(base_url: str, session_id: str, dataset_path: Path) -> dict[str, Any]:
    boundary = f"----codex-{int(time.time() * 1000)}"
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
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _chat(base_url: str, session_id: str, dataset_id: str, message: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url}/api/sessions/{session_id}/chat/stream",
        data=json.dumps({"message": message, "active_dataset_id": dataset_id}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    events: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
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
        raise RuntimeError(exc.read().decode("utf-8")) from exc
    return events


def _parse_sse_block(lines: list[str]) -> list[dict[str, Any]]:
    events = []
    for line in lines:
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _run_checks(
    expects: list[str],
    final: dict[str, Any],
    artifacts: list[dict[str, Any]],
    events: list[dict[str, Any]],
    confirmation: dict[str, Any],
) -> dict[str, bool]:
    answer = str(final.get("answer") or "").lower()
    public_trace = "\n".join(str(event.get("message") or "") for event in events if event.get("type") == "trace")
    checks: dict[str, bool] = {}
    for expectation in expects:
        if expectation == "no_step_budget":
            checks[expectation] = "step budget" not in answer
        elif expectation == "mentions_patent_metadata":
            checks[expectation] = any(term in answer for term in ("patent", "metadata", "record"))
        elif expectation == "state_changed_false":
            checks[expectation] = final.get("state_changed") is False
        elif expectation == "table_artifact":
            checks[expectation] = any(artifact.get("kind") == "table" for artifact in artifacts)
        elif expectation == "chart_artifact":
            checks[expectation] = any(artifact.get("kind") == "chart" for artifact in artifacts)
        elif expectation == "chart_data_valid":
            checks[expectation] = _has_valid_chart_data(artifacts)
        elif expectation == "csv_artifact":
            checks[expectation] = any(artifact.get("kind") == "csv" for artifact in artifacts)
        elif expectation == "no_wrapper_columns":
            checks[expectation] = not _has_wrapper_columns(artifacts)
        elif expectation == "schema_terms":
            checks[expectation] = all(term in answer for term in ("scalar", "date", "list"))
        elif expectation == "no_raw_dict_preview":
            checks[expectation] = "{'object_type':" not in answer and "... truncated ..." not in answer
        elif expectation == "no_public_verifier_exception":
            checks[expectation] = "JSONDecodeError" not in public_trace and "ValidationError" not in public_trace
        elif expectation == "clarification_answer":
            checks[expectation] = "please choose the rule" in answer
        elif expectation == "no_python_execution":
            checks[expectation] = not any(event.get("type") == "code_started" for event in events)
        elif expectation == "history_answer":
            checks[expectation] = "current branch" in answer or "original upload" in answer
        elif expectation == "no_name_error":
            checks[expectation] = "nameerror" not in answer and not any(
                "NameError" in str(event.get("traceback") or "")
                for event in events
                if event.get("type") == "code_result_summary"
            )
        elif expectation == "confirmation_required":
            checks[expectation] = bool(confirmation)
        elif expectation == "confirmation_payload_summary":
            checks[expectation] = bool(
                confirmation.get("operation_summary")
                and confirmation.get("proposed_code")
                and confirmation.get("state_impact")
                and confirmation.get("rollback_note")
            )
        else:
            checks[expectation] = False
    return checks


def _has_wrapper_columns(artifacts: list[dict[str, Any]]) -> bool:
    for artifact in artifacts:
        columns = artifact.get("columns") or artifact.get("metadata", {}).get("columns") or []
        for column in columns:
            key = column.get("key") if isinstance(column, dict) else column
            if str(key) in FORBIDDEN_COLUMNS:
                return True
    return False


def _has_valid_chart_data(artifacts: list[dict[str, Any]]) -> bool:
    for artifact in artifacts:
        if artifact.get("kind") != "chart":
            continue
        spec = artifact.get("chart_spec") or artifact.get("metadata", {}).get("chart_spec") or {}
        data = spec.get("data") if isinstance(spec, dict) else None
        x = spec.get("x") if isinstance(spec, dict) else None
        y = spec.get("y") if isinstance(spec, dict) else None
        if isinstance(data, list) and data and isinstance(data[0], dict) and x in data[0] and y in data[0]:
            return True
    return False


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Data Analysis Agent Eval Report",
        "",
        f"Passed: **{report['passed']}**",
        f"Dataset: `{report['dataset']}`",
        "",
    ]
    for result in report["results"]:
        lines.extend(
            [
                f"## {result['id']}",
                f"Passed: **{result['passed']}**",
                "",
                "Checks:",
                *[f"- {name}: {value}" for name, value in result["checks"].items()],
                "",
                "Artifacts:",
                *[f"- {artifact['kind']}: {artifact['title']}" for artifact in result["artifacts"]],
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

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
            "no_raw_type_attrs",
            "mentions_24410",
            "mentions_representative_fields",
        ],
    },
    {
        "id": "tabular_preview_not_top_countries",
        "prompt": "Convert this dataset into a tabular preview if possible. Show the inferred columns and the first 5 rows.",
        "expects": [
            "table_artifact",
            "no_wrapper_columns",
            "preview_table_domain_columns",
            "table_not_top_countries",
            "state_changed_false",
        ],
    },
    {
        "id": "schema",
        "prompt": "Summarize the schema of this patent metadata dataset. Which fields are scalar fields, which are dates, and which are list-like fields?",
        "expects": ["schema_terms", "schema_domain_fields", "no_pydantic_in_answer", "no_step_budget"],
    },
    {
        "id": "top_countries",
        "prompt": "Show the top countries by number of patent records. Return both a table and a short explanation.",
        "expects": [
            "table_artifact",
            "table_row_count_at_least_3",
            "table_counts_full_24410",
            "not_cn19_ca1_only",
            "state_changed_false",
        ],
    },
    {
        "id": "filing_date_range_no_duplicate_tables",
        "prompt": "What is the filing date range of this portfolio? Also summarize the number of filings by year in a table.",
        "expects": [
            "no_step_budget",
            "max_execute_python_calls_2",
            "table_artifact",
            "exactly_one_filings_by_year_table",
            "no_sample_filings_table",
            "filing_years_full_data",
            "filing_date_range_in_answer",
            "table_not_top_countries",
            "no_duplicate_artifact_titles",
            "state_changed_false",
        ],
    },
    {
        "id": "chart_render_contract",
        "prompt": "Create a bar chart showing the number of patent records by country.",
        "expects": ["chart_artifact", "chart_data_valid"],
    },
    {
        "id": "line_or_bar_time_series_prefers_line",
        "prompt": "Create a line chart or bar chart showing filings by year based on filing_date.",
        "expects": ["chart_artifact", "chart_data_valid", "chart_type_line", "filings_chart_fields", "chart_title_not_fake"],
    },
    {
        "id": "line_chart_filings_by_year",
        "prompt": "Create a line chart showing filings by year based on filing_date.",
        "expects": ["chart_artifact", "chart_data_valid", "chart_type_line", "filings_chart_fields", "chart_title_not_fake"],
    },
    {
        "id": "pie_chart_no_result_nameerror",
        "prompt": "create a pie chart for country",
        "expects": ["chart_artifact", "chart_data_valid", "chart_type_pie", "country_chart_fields", "no_result_nameerror"],
    },
    {
        "id": "conceptual_uses_semantic_verifier",
        "prompt": "What's in this file?",
        "expects": [
            "mentions_patent_metadata",
            "mentions_record_meaning",
            "no_raw_dict_preview",
            "no_llm_skip_trace",
            "state_changed_false",
        ],
    },
    {
        "id": "preview_metadata",
        "prompt": "Convert this dataset into a tabular preview if possible. Show the inferred columns and the first 5 rows.",
        "expects": ["table_artifact", "preview_row_count_5", "preview_source_total_24410", "state_changed_false"],
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
        "id": "delete_last_500",
        "prompt": "delete last 500 entries",
        "expects": ["confirmation_required", "delete_last_500_confirmation", "state_not_changed_before_approval"],
    },
    {
        "id": "delete_empty_title",
        "prompt": "delete empty title",
        "expects": ["full_scan_delete_empty_title", "state_not_changed_or_confirmation"],
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
        artifact_contents = {
            artifact["id"]: _artifact_content(base_url, session_id, str(artifact["id"]))
            for artifact in artifacts
            if artifact.get("id")
        }
        checks = _run_checks(question["expects"], final, artifacts, artifact_contents, events, confirmation)
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
                        "content": artifact_contents.get(artifact.get("id")),
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


def _artifact_content(base_url: str, session_id: str, artifact_id: str) -> Any:
    request = urllib.request.Request(
        f"{base_url}/api/sessions/{session_id}/artifacts/{artifact_id}/content",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError:
        return None

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


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
    artifact_contents: dict[str, Any],
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
            checks[expectation] = _has_valid_chart_data(artifacts, artifact_contents)
        elif expectation == "chart_type_line":
            spec = _first_chart_spec(artifacts, artifact_contents)
            checks[expectation] = str(spec.get("chart_type") or "").lower() == "line"
        elif expectation == "chart_type_pie":
            spec = _first_chart_spec(artifacts, artifact_contents)
            checks[expectation] = str(spec.get("chart_type") or "").lower() == "pie"
        elif expectation == "filings_chart_fields":
            spec = _first_chart_spec(artifacts, artifact_contents)
            x = str(spec.get("x") or "").lower()
            y = str(spec.get("y") or "").lower()
            checks[expectation] = x in {"filing_year", "year"} and y in {"filing_count", "count", "record_count"}
        elif expectation == "country_chart_fields":
            spec = _first_chart_spec(artifacts, artifact_contents)
            checks[expectation] = str(spec.get("x") or "").lower() == "country"
        elif expectation == "chart_title_not_fake":
            checks[expectation] = "fake agent chart" not in json.dumps(artifacts, ensure_ascii=False).lower()
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
        elif expectation == "no_raw_type_attrs":
            checks[expectation] = "'attrs'" not in answer and "{'type':" not in answer and "__pydantic" not in answer
        elif expectation == "mentions_24410":
            checks[expectation] = "24410" in answer.replace(",", "") or "24,410" in answer
        elif expectation == "mentions_representative_fields":
            checks[expectation] = sum(
                field in answer
                for field in ("country", "doc_number", "title", "owners", "assignees", "filing_date", "status")
            ) >= 4
        elif expectation == "mentions_record_meaning":
            checks[expectation] = "record represents" in answer or "each record" in answer
        elif expectation == "schema_domain_fields":
            checks[expectation] = ("owners" in answer or "assignees" in answer) and "filing_date" in answer
        elif expectation == "no_pydantic_in_answer":
            checks[expectation] = "__pydantic" not in answer and "__dict__" not in answer
        elif expectation == "table_row_count_at_least_3":
            checks[expectation] = _first_table_row_count(artifacts, artifact_contents) >= 3
        elif expectation == "preview_table_domain_columns":
            checks[expectation] = _preview_table_has_domain_columns(artifacts, artifact_contents)
        elif expectation == "table_not_top_countries":
            checks[expectation] = not _first_table_is_top_countries(artifacts, artifact_contents)
        elif expectation == "table_counts_full_24410":
            checks[expectation] = (
                _table_analyzed_row_count(artifacts, artifact_contents) in {24_410, 24410}
                or _table_count_sum(
                    artifacts,
                    artifact_contents,
                    {"count", "record_count", "patent_count", "patent_record_count"},
                )
                in {24_410, 24410}
            )
        elif expectation == "not_cn19_ca1_only":
            checks[expectation] = not _looks_like_sample_only_country_result(artifacts, artifact_contents)
        elif expectation == "filing_years_full_data":
            checks[expectation] = (
                (
                    _table_analyzed_row_count(artifacts, artifact_contents) in {24_410, 24410}
                    or _table_count_sum(artifacts, artifact_contents, {"filing_count", "count", "record_count"})
                    in {24_410, 24410}
                )
                and _first_table_row_count(artifacts, artifact_contents) > 3
            )
        elif expectation == "exactly_one_filings_by_year_table":
            checks[expectation] = len(_filings_by_year_tables(artifacts, artifact_contents)) == 1
        elif expectation == "no_sample_filings_table":
            checks[expectation] = not any(
                "sample" in str(document.get("title") or "").lower()
                for document in _filings_by_year_tables(artifacts, artifact_contents)
            )
        elif expectation == "no_duplicate_artifact_titles":
            titles = [str(artifact.get("title") or artifact.get("name") or "").lower() for artifact in artifacts]
            checks[expectation] = len(titles) == len(set(titles))
        elif expectation == "max_execute_python_calls_2":
            checks[expectation] = sum(1 for event in events if event.get("type") == "code_started") <= 2
        elif expectation == "filing_date_range_in_answer":
            checks[expectation] = "filing" in answer and ("range" in answer or "to" in answer) and any(
                str(year) in answer for year in range(2000, 2027)
            )
        elif expectation == "preview_row_count_5":
            checks[expectation] = any(
                _numeric(document.get("preview_row_count")) == 5 for document in _table_documents(artifacts, artifact_contents)
            )
        elif expectation == "preview_source_total_24410":
            checks[expectation] = any(
                _numeric(document.get("source_total_row_count") or document.get("source_row_count")) in {24_410, 24410}
                for document in _table_documents(artifacts, artifact_contents)
            ) or _first_table_row_count(artifacts, artifact_contents) == 5
        elif expectation == "no_llm_skip_trace":
            checks[expectation] = "Skipping LLM verifier" not in public_trace
        elif expectation == "no_result_nameerror":
            checks[expectation] = "NameError: name 'RESULT' is not defined" not in json.dumps(events)
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
        elif expectation == "delete_last_500_confirmation":
            text = json.dumps(confirmation, ensure_ascii=False).lower()
            checks[expectation] = "last 500" in text and "23910" in text.replace(",", "")
        elif expectation == "state_not_changed_before_approval":
            checks[expectation] = not any(event.get("state_changed") is True for event in events)
        elif expectation == "full_scan_delete_empty_title":
            checks[expectation] = bool(confirmation) or ("full" in answer and "scan" in answer and "0" in answer)
        elif expectation == "state_not_changed_or_confirmation":
            checks[expectation] = bool(confirmation) or final.get("state_changed") is False
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


def _has_valid_chart_data(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> bool:
    for artifact in artifacts:
        if artifact.get("kind") != "chart":
            continue
        content = artifact_contents.get(artifact.get("id"))
        spec = (
            artifact.get("chart_spec")
            or artifact.get("metadata", {}).get("chart_spec")
            or (content.get("chart_spec") if isinstance(content, dict) else None)
            or {}
        )
        data = spec.get("data") if isinstance(spec, dict) else None
        x = spec.get("x") if isinstance(spec, dict) else None
        y = spec.get("y") if isinstance(spec, dict) else None
        if isinstance(data, list) and data and isinstance(data[0], dict) and x in data[0] and y in data[0]:
            return True
    return False


def _first_chart_spec(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> dict[str, Any]:
    for artifact in artifacts:
        if artifact.get("kind") != "chart":
            continue
        content = artifact_contents.get(artifact.get("id"))
        spec = (
            artifact.get("chart_spec")
            or artifact.get("metadata", {}).get("chart_spec")
            or (content.get("chart_spec") if isinstance(content, dict) else None)
        )
        return spec if isinstance(spec, dict) else {}
    return {}


def _table_documents(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("kind") != "table":
            continue
        content = artifact_contents.get(artifact.get("id"))
        if isinstance(content, dict):
            documents.append(content)
    return documents


def _first_table_row_count(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> int:
    documents = _table_documents(artifacts, artifact_contents)
    if not documents:
        return 0
    rows = documents[0].get("rows")
    if isinstance(rows, list):
        return len(rows)
    row_count = documents[0].get("row_count")
    return int(row_count) if isinstance(row_count, (int, float)) and not isinstance(row_count, bool) else 0


def _table_analyzed_row_count(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> int | None:
    for document in _table_documents(artifacts, artifact_contents):
        for key in ("analyzed_row_count", "source_row_count"):
            value = document.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
    return None


def _table_count_sum(
    artifacts: list[dict[str, Any]],
    artifact_contents: dict[str, Any],
    count_keys: set[str],
) -> int | None:
    for document in _table_documents(artifacts, artifact_contents):
        rows = document.get("rows")
        if not isinstance(rows, list):
            continue
        total = 0
        found = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in count_keys:
                value = row.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    total += int(value)
                    found = True
                    break
        if found:
            return total
    return None


def _numeric(value: Any) -> int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _preview_table_has_domain_columns(
    artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]
) -> bool:
    documents = _table_documents(artifacts, artifact_contents)
    if not documents:
        return False
    keys = _table_keys(documents[0])
    required = {"country", "doc_number", "title", "owners", "assignees", "filing_date", "status"}
    return len(keys & required) >= 5


def _first_table_is_top_countries(artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]) -> bool:
    documents = _table_documents(artifacts, artifact_contents)
    if not documents:
        return False
    first = documents[0]
    title = str(first.get("title") or "").lower()
    keys = _table_keys(first)
    return "top countr" in title or (keys <= {"country", "count", "record_count", "patent_count"} and "country" in keys)


def _filings_by_year_tables(
    artifacts: list[dict[str, Any]], artifact_contents: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for document in _table_documents(artifacts, artifact_contents):
        keys = _table_keys(document)
        if keys & {"filing_year", "year"} and keys & {"filing_count", "count", "record_count"}:
            output.append(document)
    return output


def _table_keys(document: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    columns = document.get("columns") or []
    if isinstance(columns, list):
        for column in columns:
            if isinstance(column, dict):
                key = column.get("key") or column.get("label")
            else:
                key = column
            if key is not None:
                keys.add(str(key).lower())
    rows = document.get("rows")
    if not keys and isinstance(rows, list):
        for row in rows[:5]:
            if isinstance(row, dict):
                keys.update(str(key).lower() for key in row)
    return keys


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
        if country_counts and sum(country_counts.values()) == 20 and _table_analyzed_row_count(artifacts, artifact_contents) == 24410:
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

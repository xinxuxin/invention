from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.services.optimized_mutations import (
    analyze_mutation_impact,
    apply_mutation_spec,
    compare_value,
    discover_filter_fields,
    discover_record_collections,
    fast_get_path,
    is_missing_value,
    parse_mutation_request,
)
from scripts.create_agent_test_datasets import (
    build_custom_sensor_fleet,
    build_mixed_dataframe_numpy_bundle,
    build_mixed_top_level_collection,
    build_nested_customer_events,
)


@dataclass
class Reading:
    metrics: dict[str, float]


@dataclass
class Root:
    readings: list[Reading]


def test_fast_get_path_reads_nested_shapes() -> None:
    assert fast_get_path({"profile": {"country": "JP"}}, "profile.country") == "JP"
    assert fast_get_path(Reading(metrics={"battery_pct": 12.5}), "metrics.battery_pct") == 12.5
    assert fast_get_path(pd.Series({"status": "paid"}), "status") == "paid"
    assert fast_get_path({"tables": {"orders": [1, 2]}}, "tables.orders") == [1, 2]
    assert fast_get_path(Root(readings=[Reading(metrics={"battery_pct": 5})]), "readings")[0].metrics["battery_pct"] == 5
    assert fast_get_path({"profile": {}}, "profile.country") is None


def test_discover_record_collections_across_generated_datasets() -> None:
    nested_paths = {candidate.path for candidate in discover_record_collections(build_nested_customer_events())}
    assert "customers" in nested_paths
    assert "customers.events" in nested_paths
    assert "customers.events.order.items" in nested_paths
    assert "customers.support_tickets" in nested_paths

    bundle_paths = {candidate.path for candidate in discover_record_collections(build_mixed_dataframe_numpy_bundle())}
    assert {"users", "orders", "daily_metrics"}.issubset(bundle_paths)

    fleet_paths = {candidate.path for candidate in discover_record_collections(build_custom_sensor_fleet())}
    assert "sensors" in fleet_paths
    assert "sensors.readings" in fleet_paths
    assert "sensors.readings.alerts" in fleet_paths

    mixed_paths = {candidate.path for candidate in discover_record_collections(build_mixed_top_level_collection())}
    assert mixed_paths


def test_parse_mutation_intents_for_common_generic_filters() -> None:
    softbank_like = [
        {"country": "JP", "doc_number": "1", "kind": "A", "status": "Granted", "title": "A", "filing_date": "2020-01-02"},
        {"country": "US", "doc_number": "2", "kind": "A", "status": "Pending", "title": "B", "filing_date": "2021-01-02"},
    ]
    assert parse_mutation_request("delete all non japan entries", softbank_like).spec.field_path == "country"
    assert parse_mutation_request("keep only granted patents", softbank_like).spec.field_path == "status"
    assert parse_mutation_request("remove records with missing title", softbank_like).spec.operator == "is_missing"
    assert parse_mutation_request("drop duplicates by country, doc_number, kind, title", softbank_like).spec.kind == "drop_duplicates"
    assert parse_mutation_request("delete first 2000 entries", softbank_like).spec.kind == "delete_first_n"
    assert parse_mutation_request("delete last 500 entries", softbank_like).spec.kind == "delete_last_n"
    assert parse_mutation_request("add filing_year based on filing_date and persist it", softbank_like).spec.kind == "add_derived_field"

    nested = build_nested_customer_events()
    enterprise = parse_mutation_request("keep only enterprise customers", nested).spec
    assert enterprise.target_path == "customers"
    assert enterprise.field_path == "segment"

    churn = parse_mutation_request("remove customers with churn_risk above 0.8", nested).spec
    assert churn.target_path == "customers"
    assert churn.field_path == "churn_risk"
    assert churn.operator == "lte"

    bundle = build_mixed_dataframe_numpy_bundle()
    orders = parse_mutation_request("remove orders with gross_revenue below 100", bundle).spec
    assert orders.target_path == "orders"
    assert orders.field_path == "gross_revenue"

    fleet = build_custom_sensor_fleet()
    battery = parse_mutation_request("remove readings with battery_pct below 20", fleet).spec
    assert battery.target_path == "sensors.readings"
    assert battery.field_path == "metrics.battery_pct"

    assert parse_mutation_request("clean this dataset", softbank_like).clarification is not None
    assert parse_mutation_request("delete all non active entries", softbank_like).clarification is not None


def test_impact_and_apply_filter_preserve_structures() -> None:
    rows = [{"country": "JP"}, {"country": "US"}, {"country": "JP"}]
    spec = parse_mutation_request("delete all non japan entries", rows).spec
    impact = analyze_mutation_impact(rows, spec)
    assert impact.current_count == 3
    assert impact.affected_count == 1
    new_rows, preview = apply_mutation_spec(rows, spec)
    assert new_rows == [{"country": "JP"}, {"country": "JP"}]
    assert preview["new_row_count"] == 2

    frame = pd.DataFrame({"status": ["paid", "refunded", "paid"], "gross_revenue": [10, 20, 30]})
    frame_spec = parse_mutation_request("drop refunded records", frame).spec
    new_frame, _ = apply_mutation_spec(frame, frame_spec)
    assert isinstance(new_frame, pd.DataFrame)
    assert new_frame["status"].tolist() == ["paid", "paid"]

    bundle = build_mixed_dataframe_numpy_bundle()
    users_before = len(bundle["users"])
    order_spec = parse_mutation_request("remove orders with gross_revenue below 100", bundle).spec
    new_bundle, order_preview = apply_mutation_spec(bundle, order_spec)
    assert len(new_bundle["users"]) == users_before
    assert len(new_bundle["orders"]) == order_preview["new_row_count"]

    fleet = build_custom_sensor_fleet()
    original_readings = sum(len(sensor.readings) for sensor in fleet.sensors)
    battery_spec = parse_mutation_request("remove readings with battery_pct below 20", fleet).spec
    new_fleet, battery_preview = apply_mutation_spec(fleet, battery_spec)
    assert sum(len(sensor.readings) for sensor in new_fleet.sensors) == battery_preview["new_row_count"]
    assert sum(len(sensor.readings) for sensor in fleet.sensors) == original_readings


def test_missing_and_comparison_helpers() -> None:
    assert is_missing_value(None)
    assert is_missing_value("")
    assert compare_value("JP", "eq", "jp")
    assert compare_value(19, "lt", 20)
    assert compare_value("enterprise", "in", ["free", "enterprise"])
    fields = discover_filter_fields(build_nested_customer_events())
    assert any(field.field_path == "segment" and "segment" in field.semantic_tags for field in fields)

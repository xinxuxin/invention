from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd


@dataclass
class SensorReading:
    sensor_id: str
    timestamp: datetime
    metrics: dict[str, float]
    alerts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Sensor:
    sensor_id: str
    site: str
    zone: str
    readings: list[SensorReading]


@dataclass
class SensorFleet:
    fleet_id: str
    generated_at: datetime
    sensors: list[Sensor]
    metadata: dict[str, Any]


def build_nested_customer_events() -> dict[str, Any]:
    countries = ["US", "GB", "JP", "CA", "DE"]
    segments = ["startup", "enterprise", "consumer"]
    customers = []
    for idx in range(1, 16):
        country = countries[idx % len(countries)]
        events = []
        for event_idx in range(1, 4):
            total = round(80 + idx * 17.5 + event_idx * 13.25, 2)
            events.append(
                {
                    "event_id": f"evt-{idx:03d}-{event_idx}",
                    "timestamp": (datetime(2025, 1, 1) + timedelta(days=idx * event_idx)).isoformat(),
                    "channel": ["web", "sales", "partner"][event_idx % 3],
                    "event_type": "purchase",
                    "order": {
                        "order_total": total,
                        "items": [
                            {
                                "sku": f"SKU-{(idx + event_idx) % 5}",
                                "name": ["Sensor", "Gateway", "Plan", "Support", "Module"][(idx + event_idx) % 5],
                                "quantity": event_idx,
                                "unit_price": round(total / max(event_idx, 1), 2),
                            }
                        ],
                    },
                    "risk_flags": [
                        {"flag_type": "payment_delay", "severity": "medium" if idx % 4 == 0 else "low"}
                    ]
                    if idx % 4 == 0
                    else [],
                }
            )
        customers.append(
            {
                "customer_id": f"cust-{idx:03d}",
                "country": country,
                "segment": segments[idx % len(segments)],
                "joined_at": (datetime(2024, 6, 1) + timedelta(days=idx * 9)).date().isoformat(),
                "churn_risk": round((idx % 10) / 10, 2),
                "events": events,
                "support_tickets": [
                    {
                        "ticket_id": f"ticket-{idx:03d}",
                        "status": "open" if idx % 5 == 0 else "closed",
                        "severity": "high" if idx % 5 == 0 else "low",
                    }
                ]
                if idx % 3 == 0 or idx % 5 == 0
                else [],
            }
        )
    return {
        "metadata": {"source": "generated", "domain": "nested customer events"},
        "customers": customers,
        "lookup_tables": {
            "segments": [{"segment": segment, "priority": i + 1} for i, segment in enumerate(segments)],
            "countries": [{"country": country, "region": "global"} for country in countries],
        },
    }


def build_mixed_dataframe_numpy_bundle() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    users = pd.DataFrame(
        {
            "user_id": np.arange(1, 501),
            "country": rng.choice(["US", "GB", "JP", "CA", "DE"], 500),
            "segment": rng.choice(["free", "pro", "enterprise"], 500),
            "signup_date": pd.date_range("2024-01-01", periods=500, freq="D"),
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": np.arange(1000, 1600),
            "user_id": rng.integers(1, 501, 600),
            "category": rng.choice(["hardware", "software", "services"], 600),
            "status": rng.choice(["paid", "refunded"], 600, p=[0.88, 0.12]),
            "gross_revenue": np.round(rng.gamma(3, 80, 600), 2),
            "ordered_at": pd.date_range("2025-01-01", periods=600, freq="12h"),
        }
    )
    daily_metrics = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=60, freq="D"),
            "active_users": rng.integers(120, 380, 60),
            "latency_ms_p95": np.round(rng.normal(280, 45, 60), 2),
            "error_rate": np.round(rng.uniform(0.002, 0.04, 60), 4),
        }
    )
    return {
        "users": users,
        "orders": orders,
        "daily_metrics": daily_metrics,
        "user_embedding_matrix": rng.normal(size=(500, 16)),
        "cohort_tensor": rng.normal(size=(5, 12, 4)),
        "metadata": {"source": "generated mixed dataframe numpy bundle"},
    }


def build_custom_sensor_fleet() -> SensorFleet:
    sensors: list[Sensor] = []
    base = datetime(2025, 3, 1)
    for sensor_idx in range(1, 9):
        readings = []
        for hour in range(0, 24, 3):
            temperature = 21 + sensor_idx * 0.7 + hour * 0.05
            vibration = 0.1 + (sensor_idx % 3) * 0.04 + (0.4 if sensor_idx in {3, 7} and hour >= 12 else 0)
            battery = 95 - sensor_idx * 4 - hour * 0.8
            alerts = []
            if vibration > 0.45:
                alerts.append({"alert_type": "high_vibration", "severity": "high", "value": round(vibration, 3)})
            if battery < 25:
                alerts.append({"alert_type": "low_battery", "severity": "medium", "value": round(battery, 2)})
            readings.append(
                SensorReading(
                    sensor_id=f"sensor-{sensor_idx:03d}",
                    timestamp=base + timedelta(hours=hour),
                    metrics={
                        "temperature_c": round(temperature, 2),
                        "vibration_g": round(vibration, 3),
                        "battery_pct": round(battery, 2),
                    },
                    alerts=alerts,
                )
            )
        sensors.append(
            Sensor(
                sensor_id=f"sensor-{sensor_idx:03d}",
                site=["Austin", "London", "Tokyo"][sensor_idx % 3],
                zone=f"Z{sensor_idx % 4}",
                readings=readings,
            )
        )
    return SensorFleet(
        fleet_id="fleet-demo-001",
        generated_at=datetime(2025, 3, 2),
        sensors=sensors,
        metadata={"source": "generated custom sensor fleet", "owner": "demo"},
    )


def build_mixed_top_level_collection() -> list[Any]:
    return [
        {"kind": "metadata", "source": "generated", "tags": ["mixed", "top-level"]},
        pd.DataFrame({"record_id": [1, 2, 3], "score": [0.8, 0.65, 0.91], "country": ["US", "GB", "JP"]}),
        np.arange(24).reshape(6, 4),
        [
            {"record_id": "a", "payload": {"country": "US", "value": 10}},
            {"record_id": "b", "payload": {"country": "CA", "value": 20}},
        ],
        ("tuple_marker", {"status": "ok", "count": 2}),
    ]


def build_all() -> dict[str, Any]:
    return {
        "nested_customer_events.pkl": build_nested_customer_events(),
        "mixed_dataframe_numpy_bundle.pkl": build_mixed_dataframe_numpy_bundle(),
        "custom_sensor_fleet.pkl": build_custom_sensor_fleet(),
        "mixed_top_level_collection.pkl": build_mixed_top_level_collection(),
    }


def write_agent_test_datasets(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, value in build_all().items():
        path = output_dir / filename
        with path.open("wb") as handle:
            cloudpickle.dump(value, handle)
        written.append(path)
    return written


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "agent_test_datasets"
    for path in write_agent_test_datasets(output_dir):
        print(path.relative_to(repo_root))


if __name__ == "__main__":
    main()

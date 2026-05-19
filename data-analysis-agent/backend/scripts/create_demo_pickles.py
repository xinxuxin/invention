from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd


GENERATED_FILENAMES = [
    "dataframe_transactions.pkl",
    "list_of_dicts_nested.pkl",
    "numpy_array.pkl",
    "custom_objects.pkl",
    "mixed_collection.pkl",
    "multi_dataset_users.pkl",
    "multi_dataset_orders.pkl",
]


@dataclass
class DemoPatentRecord:
    patent_id: str
    owner: str
    scores: dict[str, float]
    tags: list[str]
    metadata: dict[str, Any]

    @property
    def primary_tag(self) -> str:
        return self.tags[0]


def build_demo_objects() -> dict[str, object]:
    transactions = pd.DataFrame(
        {
            "transaction_id": pd.Series([1001, 1002, 1003, 1004, 1005, 1006], dtype="int64"),
            "user_id": pd.Series([1, 2, 2, 3, 4, 5], dtype="int64"),
            "amount": [49.99, 1200.00, np.nan, 18.25, 250.50, 87.10],
            "category": pd.Categorical(["software", "hardware", "software", "services", None, "hardware"]),
            "purchased_at": pd.to_datetime(
                ["2026-01-02", "2026-01-05", None, "2026-02-14", "2026-02-20", "2026-03-01"]
            ),
            "is_refund": [False, False, True, False, False, True],
            "notes": ["annual", None, "manual review", "trial", "priority", ""],
        }
    )

    nested_records = [
        {
            "id": "alpha",
            "profile": {"name": "Alpha Labs", "region": "US", "tags": ["ai", "robotics"]},
            "metrics": {"revenue": 1200, "confidence": 0.91},
        },
        {
            "id": "beta",
            "profile": {"name": "Beta KK", "address": {"country": "JP", "city": "Tokyo"}},
            "events": [{"type": "filed", "year": 2024}, {"type": "granted", "year": 2026}],
        },
        {
            "id": "gamma",
            "metrics": {"revenue": None, "confidence": 0.64, "owners": ["SoftBank", "Arm"]},
            "extra": {"notes": {"source": "demo", "reviewed": True}},
        },
        {"id": "delta", "profile": None, "flags": ["missing-metrics"]},
    ]

    numeric_array = np.array(
        [
            [0.15, 1.2, 3.4, np.nan],
            [2.2, 5.1, 8.0, 13.0],
            [21.0, 34.0, 55.0, 89.0],
        ],
        dtype="float64",
    )

    custom_objects = [
        DemoPatentRecord(
            patent_id="US-2026-0001",
            owner="SoftBank Group",
            scores={"novelty": 0.82, "market_fit": 0.76},
            tags=["telecom", "edge"],
            metadata={"jurisdictions": ["US", "JP"], "citations": 14},
        ),
        DemoPatentRecord(
            patent_id="JP-2026-0002",
            owner="Arm",
            scores={"novelty": 0.91, "market_fit": 0.88},
            tags=["semiconductor", "ai"],
            metadata={"jurisdictions": ["JP", "EP"], "citations": 31},
        ),
        DemoPatentRecord(
            patent_id="EP-2026-0003",
            owner="PayPay",
            scores={"novelty": 0.69, "market_fit": 0.8},
            tags=["payments", "security"],
            metadata={"jurisdictions": ["EP"], "citations": 8},
        ),
    ]

    mixed_collection = {
        "frame": transactions.head(3).copy(),
        "records": nested_records,
        "matrix": numeric_array,
        "scalar": 42,
        "nested": {
            "objects": custom_objects[:2],
            "status": {"loaded": True, "source": "generated demo"},
        },
    }

    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "email": ["a@example.com", "b@example.com", None, "d@example.com"],
            "segment": ["startup", "enterprise", "startup", "public-sector"],
            "joined_at": pd.to_datetime(["2025-11-01", "2025-12-15", "2026-01-20", "2026-02-01"]),
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [501, 502, 503, 504, 505],
            "user_id": [1, 2, 2, 4, 99],
            "amount": [120.0, 950.0, 1250.0, 300.0, 75.0],
            "ordered_at": pd.to_datetime(["2026-01-10", "2026-01-12", "2026-02-02", "2026-02-09", "2026-03-01"]),
        }
    )

    return {
        "dataframe_transactions.pkl": transactions,
        "list_of_dicts_nested.pkl": nested_records,
        "numpy_array.pkl": numeric_array,
        "custom_objects.pkl": custom_objects,
        "mixed_collection.pkl": mixed_collection,
        "multi_dataset_users.pkl": users,
        "multi_dataset_orders.pkl": orders,
    }


def write_demo_pickles(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, value in build_demo_objects().items():
        path = output_dir / filename
        with path.open("wb") as target:
            cloudpickle.dump(value, target)
        written.append(path)
    return written


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    output_dir = backend_root / "tests" / "fixtures" / "generated"
    written = write_demo_pickles(output_dir)
    for path in written:
        print(path.relative_to(backend_root))


if __name__ == "__main__":
    main()

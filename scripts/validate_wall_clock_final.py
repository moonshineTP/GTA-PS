"""Validate and extract a reference wall-clock host artifact."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


METHODS = ("standard_ps", "gta_beta")
TABLE_METRICS = (
    "recovery_auc",
    "recovery_auc_linf",
    "shift_end_l1",
    "shift_end_linf",
    "mean_completed_backups_per_second",
)


def _ci95(values: np.ndarray) -> float:
    return float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size))


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--host", choices=("dp", "dyna"), required=True)
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    raw = payload["raw"]

    assert protocol["host"] == args.host
    assert protocol["n_states"] == 200
    assert protocol["reward_support"] == 20
    assert protocol["n_shifts"] == 6
    assert protocol["period_seconds"] == 5.0
    assert protocol["backup_budget_per_call"] == 10
    assert protocol["seeds"] == list(range(100))
    assert protocol["workers"] == 8
    assert [row["seed"] for row in raw] == list(range(100))

    for row in raw:
        assert row["n_states"] == 200
        assert set(row["records"]) == set(METHODS)
        for method in METHODS:
            record = row["records"][method]
            assert len(record["periods"]) == 6
            for shift, period in enumerate(record["periods"]):
                assert period["shift"] == shift
                assert period["duration_seconds"] >= 5.0
                assert period["completed_calls"] >= 100
                assert period["requested_backups"] == period["completed_calls"] * 10
                assert all(
                    math.isfinite(float(period[key]))
                    for key in (
                        "duration_seconds",
                        "recovery_auc",
                        "recovery_auc_linf",
                        "shift_end_l1",
                        "shift_end_linf",
                    )
                )
            for metric in TABLE_METRICS:
                assert math.isfinite(float(record[metric]))

    for method in METHODS:
        for metric, aggregate in payload["summary"][method].items():
            if not isinstance(aggregate, dict) or set(("mean", "ci95")) - set(aggregate):
                continue
            values = np.asarray(
                [row["records"][method][metric] for row in raw],
                dtype=float,
            )
            assert _close(aggregate["mean"], float(values.mean()))
            assert _close(aggregate["ci95"], _ci95(values))

    for metric, aggregate in payload["paired"].items():
        differences = np.asarray(
            [
                row["records"]["gta_beta"][metric]
                - row["records"]["standard_ps"][metric]
                for row in raw
            ],
            dtype=float,
        )
        assert _close(
            aggregate["mean_difference_gta_minus_standard"],
            float(differences.mean()),
        )
        assert _close(aggregate["ci95"], _ci95(differences))

    for suffix in (".pdf", ".png"):
        assert args.artifact.with_suffix(suffix).is_file()

    for method in METHODS:
        cells = []
        for metric in TABLE_METRICS:
            aggregate = payload["summary"][method][metric]
            if metric == "mean_completed_backups_per_second":
                cells.append(f"{aggregate['mean']:.0f}+/-{aggregate['ci95']:.0f}")
            else:
                cells.append(f"{aggregate['mean']:.3f}+/-{aggregate['ci95']:.3f}")
        print(method, *cells)
    print("validation=passed")


if __name__ == "__main__":
    main()

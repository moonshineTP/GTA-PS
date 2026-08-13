"""Validate and extract the reference solver-scaling artifact."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


SIZES = (400, 600, 650, 700, 750, 800, 850, 900)
MODES = ("lu_woodbury", "gmres")


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    measurements = payload["measurements"]

    assert protocol["candidate_sizes"] == list(SIZES)
    assert protocol["seeds"] == list(range(100))
    assert protocol["workers"] == 5
    assert protocol["repeats"] == 3
    assert protocol["rank_k"] == 8
    assert set(map(int, measurements)) == set(SIZES)

    for size in SIZES:
        record = measurements[str(size)]
        seed_records = record["seed_records"]
        assert [item["seed"] for item in seed_records] == list(range(100))
        for item in seed_records:
            assert set(MODES).issubset(item)
            assert all(math.isfinite(float(item[mode])) for mode in MODES)
            assert all(
                math.isfinite(float(value))
                for value in item["calibration"].values()
            )
        for mode in MODES:
            values = np.asarray([item[mode] for item in seed_records], dtype=float)
            mean = float(np.mean(values))
            median = float(np.median(values))
            std = float(np.std(values, ddof=1))
            ci95 = float(student_t.ppf(0.975, 99) * std / np.sqrt(100))
            aggregate = record[mode]
            assert aggregate["seed_samples"] == 100
            assert aggregate["repeats_per_seed"] == 3
            assert _close(aggregate["mean_seconds"], mean)
            assert _close(aggregate["median_seconds"], median)
            assert _close(aggregate["std_seconds"], std)
            assert _close(aggregate["ci95_seconds"], ci95)

    expected_crossover = min(
        SIZES,
        key=lambda size: abs(measurements[str(size)]["log_time_ratio"]),
    )
    assert payload["crossover_candidate"] == expected_crossover
    for suffix in (".pdf", ".png"):
        assert args.artifact.with_suffix(suffix).is_file()

    for size in SIZES:
        record = measurements[str(size)]
        lu = record["lu_woodbury"]
        gmres = record["gmres"]
        print(
            size,
            f"{lu['mean_seconds'] * 1e3:.2f}+/-{lu['ci95_seconds'] * 1e3:.2f}",
            f"{gmres['mean_seconds'] * 1e3:.2f}+/-"
            f"{gmres['ci95_seconds'] * 1e3:.2f}",
        )
    print(f"crossover_candidate={expected_crossover}")
    print("validation=passed")


if __name__ == "__main__":
    main()

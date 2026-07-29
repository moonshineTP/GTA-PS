"""Validate and extract the final RQ7 table-pass artifact."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


WIDTHS = (1, 2, 4, 8, 16)
ABSTRACTION_SIZES = (24, 40, 64)
METRICS = {"first_plan_l1", "shift_end_l1"}


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    seed_results = payload["seed_results"]
    results = payload["results"]

    assert protocol["states"] == 104
    assert protocol["room_size"] == 5
    assert protocol["k"] == 2
    assert protocol["gamma"] == 0.95
    assert protocol["budget"] == 20
    assert protocol["n_shifts"] == 6
    assert protocol["steps_per_shift"] == 25
    assert tuple(protocol["seeds"]) == tuple(range(100))
    assert protocol["workers"] == 6
    assert tuple(protocol["batch_sizes"]) == WIDTHS
    assert tuple(protocol["abstraction_sizes"]) == ABSTRACTION_SIZES
    assert protocol["abstraction_batch_sizes"] == [1]
    assert protocol["plot_metric"] == "shift_end_l1"
    assert _close(protocol["gps_upsilon"], 0.05)

    seed_ids = [entry["seed"] for entry in seed_results]
    assert seed_ids == list(range(100))
    assert len(set(seed_ids)) == 100

    scheduler_methods = {
        f"{prefix}{width}"
        for prefix in (
            "standard_ps_batch_",
            "gps_h2_block_",
            "gta_beta_batch_",
        )
        for width in WIDTHS
    }
    abstraction_methods = {
        f"{prefix}_a_{size}_batch_1"
        for prefix in ("vast_vi", "gta_ps_fa")
        for size in ABSTRACTION_SIZES
    }
    expected_methods = scheduler_methods | abstraction_methods
    assert set(results) == expected_methods

    for entry in seed_results:
        rewards = np.asarray(entry["reward_sequence"], dtype=float)
        assert rewards.shape == (6, 104)
        assert np.isfinite(rewards).all()
        assert set(entry["metrics"]) == expected_methods
        for method_metrics in entry["metrics"].values():
            assert set(method_metrics) == METRICS
            for values in method_metrics.values():
                array = np.asarray(values, dtype=float)
                assert array.shape == (6,)
                assert np.isfinite(array).all()

    for method in expected_methods:
        for metric in METRICS:
            per_seed = np.asarray(
                [
                    np.mean(entry["metrics"][method][metric])
                    for entry in seed_results
                ],
                dtype=float,
            )
            mean = float(np.mean(per_seed))
            std = float(np.std(per_seed, ddof=1))
            ci95 = float(student_t.ppf(0.975, 99) * std / np.sqrt(100))
            aggregate = results[method][metric]
            assert aggregate["n"] == 100
            assert _close(aggregate["mean"], mean)
            assert _close(aggregate["std"], std)
            assert _close(aggregate["ci95"], ci95)

    for suffix in (".pdf", ".png"):
        assert args.artifact.with_suffix(suffix).is_file()

    print("scheduler")
    for width in WIDTHS:
        cells = []
        for prefix in (
            "standard_ps_batch_",
            "gps_h2_block_",
            "gta_beta_batch_",
        ):
            row = results[f"{prefix}{width}"]["shift_end_l1"]
            cells.append(f"{row['mean']:.3f}+/-{row['ci95']:.3f}")
        print(width, *cells)

    print("abstraction")
    for size in ABSTRACTION_SIZES:
        standard = results[f"vast_vi_a_{size}_batch_1"]["shift_end_l1"]
        gta = results[f"gta_ps_fa_a_{size}_batch_1"]["shift_end_l1"]
        reduction = 100.0 * (standard["mean"] - gta["mean"]) / standard["mean"]
        print(
            size,
            f"{standard['mean']:.3f}+/-{standard['ci95']:.3f}",
            f"{gta['mean']:.3f}+/-{gta['ci95']:.3f}",
            f"{reduction:.1f}%",
        )

    adverse = []
    for width in WIDTHS:
        gta = results[f"gta_beta_batch_{width}"]["shift_end_l1"]["mean"]
        for comparator, prefix in (
            ("Standard-PS", "standard_ps_batch_"),
            ("GPS-H2", "gps_h2_block_"),
        ):
            other = results[f"{prefix}{width}"]["shift_end_l1"]["mean"]
            if gta > other:
                adverse.append((f"m={width}", comparator, gta - other))
    for size in ABSTRACTION_SIZES:
        gta = results[f"gta_ps_fa_a_{size}_batch_1"]["shift_end_l1"]["mean"]
        standard = results[f"vast_vi_a_{size}_batch_1"]["shift_end_l1"]["mean"]
        if gta > standard:
            adverse.append((f"a={size}", "Std-PS-FA", gta - standard))

    if adverse:
        print(
            "WARNING: Fresh final-pass data contradicts at least one fixed "
            "interpretation sentence in neurips/rebuttal/final.md."
        )
        for row, comparator, difference in adverse:
            print(f"{row}: GTA minus {comparator} = {difference:+.6f}")
    else:
        print("adverse_rows=none")

    print("validation=passed")


if __name__ == "__main__":
    main()

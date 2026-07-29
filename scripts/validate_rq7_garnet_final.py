"""Validate the final GARNET-75 GPS-H2 versus GTA-PS table artifact."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


BATCH_SIZES = (1, 2, 4, 8, 16)
METRICS = ("first_plan_l1", "shift_end_l1")


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    seed_results = payload["seed_results"]
    aggregate = payload["results"]

    expected_protocol = {
        "host": "oracle_dp",
        "env": "GarnetEnv",
        "states": 75,
        "actions": 4,
        "branching": 3,
        "n_nbr_rate": 0.15,
        "reward_support": 3,
        "reward_locality": True,
        "gamma": 0.95,
        "budget": 20,
        "n_shifts": 6,
        "steps_per_shift": 25,
        "workers": 8,
        "batch_sizes": list(BATCH_SIZES),
    }
    for key, expected in expected_protocol.items():
        assert protocol[key] == expected, (key, protocol[key], expected)
    assert protocol["seeds"] == list(range(100))
    assert protocol["gta_top_q"] is None
    assert protocol["gta_beta_parameters"] == {
        "alpha": 0.455302,
        "beta1": 0.346922,
        "beta2": 0.269094,
        "eta": 0.015734,
        "b": 3.82617,
        "arnoldi_k": 15,
    }

    expected_methods = {
        f"{prefix}{size}"
        for prefix in ("gps_h2_block_", "gta_beta_batch_")
        for size in BATCH_SIZES
    }
    assert [entry["seed"] for entry in seed_results] == list(range(100))
    assert set(aggregate) == expected_methods

    for entry in seed_results:
        rewards = np.asarray(entry["reward_sequence"], dtype=float)
        assert rewards.shape == (6, 75)
        assert np.isfinite(rewards).all()
        assert np.count_nonzero(rewards, axis=1).tolist() == [3] * 6
        assert set(entry["metrics"]) == expected_methods
        for method_metrics in entry["metrics"].values():
            assert set(method_metrics) == set(METRICS)
            for values in method_metrics.values():
                array = np.asarray(values, dtype=float)
                assert array.shape == (6,)
                assert np.isfinite(array).all()

    for method in sorted(expected_methods):
        for metric in METRICS:
            per_seed = np.asarray(
                [
                    np.mean(entry["metrics"][method][metric])
                    for entry in seed_results
                ],
                dtype=float,
            )
            sample_std = float(np.std(per_seed, ddof=1))
            ci95 = (
                float(student_t.ppf(0.975, df=per_seed.size - 1))
                * sample_std
                / np.sqrt(per_seed.size)
            )
            saved = aggregate[method][metric]
            assert saved["n"] == 100
            assert _close(saved["mean"], float(np.mean(per_seed)))
            assert _close(saved["std"], sample_std)
            assert _close(saved["ci95"], ci95)

    for suffix in (".pdf", ".png"):
        plot_path = args.artifact.with_suffix(suffix)
        assert plot_path.is_file() and plot_path.stat().st_size > 0

    for size in BATCH_SIZES:
        gps = aggregate[f"gps_h2_block_{size}"]["shift_end_l1"]
        gta = aggregate[f"gta_beta_batch_{size}"]["shift_end_l1"]
        print(
            f"{size} "
            f"{gps['mean']:.3f}+/-{gps['ci95']:.3f} "
            f"{gta['mean']:.3f}+/-{gta['ci95']:.3f}"
        )
    print("validation=passed")


if __name__ == "__main__":
    main()

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "misc"))
sys.path.insert(0, str(ROOT / "src"))

rq8_time = importlib.import_module("rq8_time_ablation")


def test_counterfactual_gains_match_direct_one_state_backup():
    P = np.array([[[0.0, 1.0], [0.0, 1.0]]])
    reward = np.array([1.0, 0.0])
    value = np.zeros(2)
    value_star = np.array([1.0, 0.0])

    gains = rq8_time.counterfactual_backup_gains(
        P, reward, 0.0, value, value_star, np.array([0, 1])
    )

    assert gains.tolist() == pytest.approx([1.0, 0.0])


def test_prefix_auc_uses_all_observed_backup_times():
    assert rq8_time.prefix_auc([1.0, 0.5, 0.25]) == pytest.approx(7.0 / 12.0)


def test_standard_queue_order_sorts_residual_descending_then_state_id():
    module = type("Module", (), {"xi": np.array([0.2, 0.4, 0.4, 0.0])})()

    assert rq8_time.standard_queue_order(module).tolist() == [1, 2, 0]


def test_summary_uses_seed_as_independent_unit_after_shift_averaging():
    rows = []
    for seed, values in ((0, (1.0, 0.5)), (1, (1.0, 0.75))):
        for shift in (0, 1):
            for backup, value in enumerate(values):
                rows.append(
                    {
                        "method": "standard_ps",
                        "seed": seed,
                        "shift": shift,
                        "backup": backup,
                        "normalized_error": value,
                        "prefix_error_auc": np.mean(values[: backup + 1]),
                        "queue_ndcg": np.nan,
                        "blocked_frontier_count": np.nan,
                        "blocked_useful_count": np.nan,
                        "blocked_queued_count": np.nan,
                        "sensitivity_median": np.nan,
                        "sensitivity_q10": np.nan,
                        "sensitivity_q90": np.nan,
                    }
                )
    for row in rows.copy():
        clone = row.copy()
        clone["method"] = "gta_beta"
        rows.append(clone)

    summary = rq8_time.summarize_traces(rows, budget=1)

    assert summary["curves"]["standard_ps"]["normalized_error"][1]["mean"] == pytest.approx(0.625)
    assert summary["cumulative_tracking_gap"][1]["mean"] == pytest.approx(0.0)

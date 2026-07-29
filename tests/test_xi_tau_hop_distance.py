import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

hop_audit = importlib.import_module("xi_tau_hop_distance")


def test_shortest_hops_follow_directed_paths_to_source():
    transition = np.array(
        [
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )

    distances = hop_audit.shortest_hops_to_source(transition, source=2)

    assert distances.tolist() == [2, 1, 0]


def test_normalized_auc_uses_trapezoidal_call_weights():
    assert hop_audit.normalized_auc([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_summarize_averages_shifts_within_seed_before_confidence_interval():
    seed_results = [
        {
            "seed": 0,
            "bins": {
                "0-2": {
                    "xi_auc": 1.0,
                    "tau_auc": 3.0,
                    "beta_tau_auc": 1.0,
                    "topology_share": 0.5,
                }
            },
        },
        {
            "seed": 1,
            "bins": {
                "0-2": {
                    "xi_auc": 3.0,
                    "tau_auc": 5.0,
                    "beta_tau_auc": 1.0,
                    "topology_share": 0.25,
                }
            },
        },
    ]

    summary = hop_audit.summarize_seed_results(seed_results)

    assert summary["0-2"]["xi_auc"]["mean"] == pytest.approx(2.0)
    assert summary["0-2"]["tau_auc"]["mean"] == pytest.approx(4.0)
    assert summary["0-2"]["topology_share"]["mean"] == pytest.approx(0.375)
    assert summary["0-2"]["xi_auc"]["n"] == 2

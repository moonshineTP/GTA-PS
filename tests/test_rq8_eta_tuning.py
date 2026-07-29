import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "misc"))
sys.path.insert(0, str(ROOT / "src"))

rq8 = importlib.import_module("rq8_eta_tuning")


def test_rq8_default_fourrooms_is_13_by_13_with_104_states():
    from gta_ps.envs.fourrooms import FourRoomsTabular

    env = FourRoomsTabular(room_size=rq8.DEFAULT_ROOM_SIZE, k=1)

    assert rq8.DEFAULT_ROOM_SIZE == 5
    assert 2 * env.room_size + env.corridor_len + 2 == 13
    assert env.n_states == 104


def test_eta_grid_contains_exact_canonical_point():
    eta_star, eta_values = rq8.resolve_eta_grid(0.95, (0.5, 1.0, 2.0))

    assert eta_star == pytest.approx((1.0 - 0.95) / 0.95)
    assert eta_values == pytest.approx(
        (0.5 * eta_star, eta_star, 2.0 * eta_star)
    )


def test_eta_grid_requires_positive_unique_multipliers_and_sr_point():
    with pytest.raises(ValueError, match="canonical"):
        rq8.resolve_eta_grid(0.95, (0.5, 2.0))
    with pytest.raises(ValueError, match="positive"):
        rq8.resolve_eta_grid(0.95, (0.0, 1.0))
    with pytest.raises(ValueError, match="unique"):
        rq8.resolve_eta_grid(0.95, (1.0, 1.0))


def test_queue_ndcg_is_one_for_oracle_order_and_lower_when_reversed():
    relevance = np.array([3.0, 2.0, 1.0, 0.0])

    assert rq8.ndcg_at_k(relevance, np.array([0, 1, 2, 3]), 3) == pytest.approx(1.0)
    assert rq8.ndcg_at_k(relevance, np.array([3, 2, 1, 0]), 3) < 1.0


def test_queue_order_follows_residual_then_topology_policy():
    module = SimpleNamespace(
        xi=np.array([0.4, 0.2, 0.0, 0.0]),
        tau=np.array([0.0, 1.0, 9.0, 2.0]),
        beta_t=1.0,
        _top_q={0, 1, 2, 3},
    )

    assert rq8.queue_order(module).tolist() == [1, 0, 2, 3]


def test_trace_snapshot_tracks_tau_scale_and_alpha_beta_gain():
    module = SimpleNamespace(
        tau=np.array([1.0, 2.0, 4.0]),
        xi=np.array([0.5, 0.5, 1.0]),
        alpha_t=0.25,
        beta_t=0.5,
        beta1=0.4,
        beta2=0.5,
        slem=0.25,
        eta=0.2,
        _tau_refresh_count=2,
        _last_tau_refresh_step=4,
        _step_count=7,
    )

    row = rq8.trace_snapshot(
        module=module,
        value=np.array([1.0, 0.0, 0.0]),
        value_star=np.zeros(3),
        initial_error=2.0,
        seed=3,
        shift=1,
        call=2,
        eta_multiplier=4.0,
        budget=5,
    )

    assert row["normalized_error"] == pytest.approx(1.0 / 6.0)
    assert row["tau_mean"] == pytest.approx(7.0 / 3.0)
    assert row["topology_gain"] == pytest.approx(0.5 * 7.0 / 2.0)
    assert row["tau_refresh_age"] == 3
    assert row["cumulative_backups"] == 7
    assert 0.0 <= row["queue_ndcg"] <= 1.0
    assert row["queue_ndcg_k"] == 3
    assert row["finite"] is True


def test_trace_snapshot_rejects_beta_schedule_mismatch():
    module = SimpleNamespace(
        tau=np.ones(2),
        xi=np.ones(2),
        alpha_t=0.25,
        beta_t=99.0,
        beta1=0.4,
        beta2=0.5,
        slem=0.25,
        eta=0.2,
        _tau_refresh_count=1,
        _last_tau_refresh_step=0,
        _step_count=0,
    )

    with pytest.raises(RuntimeError, match="beta schedule"):
        rq8.trace_snapshot(
            module=module,
            value=np.zeros(2),
            value_star=np.zeros(2),
            initial_error=1.0,
            seed=0,
            shift=0,
            call=0,
            eta_multiplier=1.0,
            budget=1,
        )


def test_eta_summary_uses_seed_as_independent_unit():
    rows = []
    paths = {
        (0, 1.0): [1.0, 0.5, 0.25],
        (1, 1.0): [1.0, 0.7, 0.4],
        (0, 2.0): [1.0, 0.4, 0.3],
        (1, 2.0): [1.0, 0.6, 0.35],
    }
    for (seed, multiplier), errors in paths.items():
        for call, error in enumerate(errors):
            rows.append(
                {
                    "seed": seed,
                    "shift": 0,
                    "call": call,
                    "cumulative_backups": call,
                    "eta_multiplier": multiplier,
                    "normalized_error": error,
                }
            )

    summary = rq8.summarize_eta_metrics(rows)

    canonical = summary[1.0]
    assert canonical["n"] == 2
    assert canonical["auc_mean"] == pytest.approx(
        np.mean([np.mean(paths[(0, 1.0)]), np.mean(paths[(1, 1.0)])])
    )
    assert canonical["roughness_mean"] == pytest.approx(0.0)


def test_paired_bootstrap_p_value_respects_finite_resample_resolution():
    rows = []
    for seed in (0, 1):
        for multiplier, end_error in ((1.0, 0.8), (2.0, 0.6)):
            for call, error in enumerate((1.0, end_error)):
                rows.append(
                    {
                        "seed": seed,
                        "shift": 0,
                        "call": call,
                        "cumulative_backups": call,
                        "eta_multiplier": multiplier,
                        "normalized_error": error,
                    }
                )

    comparison = rq8.paired_auc_comparisons(
        rows,
        bootstrap_resamples=100,
    )[0]

    assert 0.0 < comparison["p_value"] <= 1.0


def test_data_payload_is_strict_json_and_table_ready():
    rows = []
    for seed in (0, 1):
        for multiplier in (1.0, 2.0):
            for call, error in enumerate((1.0, 0.5)):
                rows.append(
                    {
                        "seed": seed,
                        "shift": 0,
                        "call": call,
                        "cumulative_backups": call,
                        "eta_multiplier": multiplier,
                        "normalized_error": error,
                        "queue_ndcg": 0.8,
                    }
                )
    summary = rq8.summarize_eta_metrics(rows, bootstrap_resamples=100)
    payload = rq8.build_data_payload(
        protocol={"eta_star": 0.05, "budget": 1},
        shifts=[],
        traces=rows,
        summary=summary,
        comparisons=[],
    )

    assert len(payload["table_rows"]) == 2
    assert len(payload["seed_rows"]) == 4
    assert len(payload["traces"]) == 8
    assert payload["table_rows"][0]["auc_mean"] == pytest.approx(0.75)
    json.dumps(payload, allow_nan=False)

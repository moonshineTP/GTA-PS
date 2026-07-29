from __future__ import annotations

import numpy as np
import pytest

from experiments import rq7_gps_batch_topq as rq7


def test_final_protocol_defaults_exclude_discarded_width() -> None:
    args = rq7.build_parser().parse_args([])

    assert tuple(args.batch_sizes) == (1, 2, 4, 8, 16)
    assert args.output_stem == "rq7_gps_batch_topq"
    assert rq7.ABSTRACTION_BATCH_SIZE == 1
    assert rq7.PLOT_METRIC == "shift_end_l1"


def test_agents_run_abstractions_only_at_m1() -> None:
    env = rq7.FourRoomsTabular(k=2, room_size=5, seed=0)

    agents = rq7._make_agents(
        env,
        gamma=0.95,
        batch_sizes=(1, 2),
        abstraction_sizes=(24,),
        gta_beta_kwargs=rq7.GTA_BETA_KWARGS,
    )

    abstraction_names = {
        name for name in agents if name.startswith(("vast_vi_", "gta_ps_fa_"))
    }
    assert abstraction_names == {
        "vast_vi_a_24_batch_1",
        "gta_ps_fa_a_24_batch_1",
    }


def test_aggregate_uses_each_seed_as_independent_unit() -> None:
    seed_results = [
        {
            "seed": seed,
            "reward_sequence": [],
            "metrics": {
                "method": {
                    "first_plan_l1": values,
                    "shift_end_l1": values,
                }
            },
        }
        for seed, values in enumerate(([1.0, 3.0], [5.0, 7.0]))
    ]

    result = rq7._aggregate(seed_results)["method"]["shift_end_l1"]

    assert result["mean"] == pytest.approx(4.0)
    assert result["std"] == pytest.approx(np.sqrt(8.0))
    assert result["n"] == 2
    assert result["ci95"] > 0.0

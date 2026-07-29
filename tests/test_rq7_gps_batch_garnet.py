from __future__ import annotations

from pathlib import Path

from experiments import rq7_gps_batch_garnet as rq7_garnet


ROOT = Path(__file__).resolve().parents[1]


def test_loads_established_garnet_s75_config() -> None:
    config = rq7_garnet.load_garnet_config(ROOT / "configs" / "garnet_s75.yaml")

    assert config["n_states"] == 75
    assert config["n_actions"] == 4
    assert config["branching"] == 3
    assert config["n_nbr_rate"] == 0.15
    assert config["reward_support"] == 3
    assert config["reward_locality"] is True
    assert config["gta_beta"] == {
        "alpha": 0.455302,
        "beta1": 0.346922,
        "beta2": 0.269094,
        "eta": 0.015734,
        "b": 3.82617,
        "arnoldi_k": 15,
    }


def test_seed_run_is_garnet_and_scheduler_only() -> None:
    config = rq7_garnet.load_garnet_config(ROOT / "configs" / "garnet_s75.yaml")
    result = rq7_garnet.run_seed(
        seed=0,
        config=config,
        gamma=0.95,
        budget=2,
        n_shifts=1,
        steps_per_shift=1,
        batch_sizes=(1, 2),
    )

    assert len(result["reward_sequence"]) == 1
    assert len(result["reward_sequence"][0]) == 75
    assert set(result["metrics"]) == {
        "gps_h2_block_1",
        "gps_h2_block_2",
        "gta_beta_batch_1",
        "gta_beta_batch_2",
    }

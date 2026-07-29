from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "misc"))
sys.path.insert(0, str(ROOT / "src"))

from gta_ps.envs.garnet import GarnetEnv  # noqa: E402
from gta_ps.algorithms import make_agent  # noqa: E402
from gta_ps.utils.hyperparams import load_tuned_for_env  # noqa: E402
from wall_clock_multishift import (  # noqa: E402
    DEFAULT_N_STATES,
    DEFAULT_REWARD_SUPPORT,
    shifted_geometric_mean,
    sparse_value_iteration,
    time_weighted_error,
)


def test_default_environment_is_sparse_garnet_200_with_moderate_reward_support() -> None:
    env = GarnetEnv(
        n_states=DEFAULT_N_STATES,
        n_actions=5,
        branching=3,
        reward_support=DEFAULT_REWARD_SUPPORT,
        n_nbr_rate=0.10,
        seed=0,
    )

    assert env.n_states == 200
    assert np.count_nonzero(env.r) == 20
    assert np.count_nonzero(env.P, axis=2).max() == 3


def test_dp_host_tracks_completed_backups_and_state_counts() -> None:
    env = GarnetEnv(n_states=20, n_actions=3, branching=3, reward_support=5, seed=2)
    agent = make_agent(
        "standard_ps",
        host="dp",
        P=env.P,
        r=env.r,
        gamma=env.gamma,
    )

    agent.plan(K=7)

    assert agent.completed_backups == 7
    assert agent.backup_counts.sum() == 7
    assert np.count_nonzero(agent.backup_counts) > 0

    agent.reset()

    assert agent.completed_backups == 0
    assert agent.backup_counts.sum() == 0


def test_dyna_host_tracks_completed_planning_backups_and_state_counts() -> None:
    env = GarnetEnv(n_states=20, n_actions=3, branching=3, reward_support=5, seed=3)
    agent = make_agent(
        "standard_ps",
        host="dyna",
        P=env.P,
        r=env.r,
        gamma=env.gamma,
        rng=np.random.default_rng(3),
    )

    agent.plan_steps(K=7)

    assert agent.completed_backups == 7
    assert agent.backup_counts.sum() == 7
    assert np.count_nonzero(agent.backup_counts) > 0
    assert agent.completed_real_updates == 0

    agent.real_step(0, 0, float(env.r[0]), 1)
    assert agent.completed_real_updates == 1

    agent.reset()

    assert agent.completed_backups == 0
    assert agent.backup_counts.sum() == 0
    assert agent.completed_real_updates == 0


def test_garnet_200_config_uses_stability_constrained_beta_eta() -> None:
    config = load_tuned_for_env("garnet_s200_dense")["gta_beta"]

    assert config["beta1"] == pytest.approx(0.1)
    assert config["eta"] == pytest.approx(0.2)
    assert config["beta1"] * (1.0 + config["beta2"]) < config["eta"]
    assert config["slem_recompute_every"] == 200


def test_sgm10_uses_ten_second_shift() -> None:
    values = np.asarray([0.0, 30.0])

    assert shifted_geometric_mean(values) == pytest.approx(10.0)


def test_time_weighted_error_normalizes_trapezoidal_area() -> None:
    times = np.asarray([0.0, 1.0, 2.0])
    errors = np.asarray([4.0, 2.0, 0.0])

    assert time_weighted_error(times, errors) == pytest.approx(2.0)


def test_time_weighted_error_rejects_invalid_trace() -> None:
    with pytest.raises(ValueError):
        time_weighted_error(np.asarray([1.0, 2.0]), np.asarray([1.0, 0.0]))


def test_sparse_oracle_matches_garnet_value_iteration() -> None:
    env = GarnetEnv(n_states=30, n_actions=3, branching=3, seed=4)

    expected = env.value_iteration(env.r)
    actual = sparse_value_iteration(
        env.P,
        env.r,
        env.gamma,
        branching=env.branching,
    )

    assert actual == pytest.approx(expected, abs=2e-3)

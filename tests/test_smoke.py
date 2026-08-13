"""
Fast smoke tests — verify all imports and basic shapes, < 30s total.
"""
import numpy as np
import pytest

from gta_ps.envs.garnet import GarnetEnv
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms.standard_ps import StandardPS
from gta_ps.algorithms.gta_ps import GtaPS
from gta_ps.algorithms.empirical_model import CountModel
from gta_ps.utils.spectral import out_laplacian, in_laplacian, slem_arnoldi, resolvent_forward
from gta_ps.utils.metrics import value_error_inf, steps_to_epsilon


# ------------------------------------------------------------------ #
# Environment tests — GARNET
# ------------------------------------------------------------------ #

def test_garnet_shapes():
    env = GarnetEnv(n_states=20, n_actions=4, branching=3, seed=0)
    assert env.P.shape == (4, 20, 20)
    assert env.r.shape == (20,)
    np.testing.assert_allclose(env.P.sum(axis=2), np.ones((4, 20)), atol=1e-10)


def test_garnet_vi():
    env = GarnetEnv(n_states=20, n_actions=4, branching=3, seed=1)
    V = env.value_iteration()
    assert V.shape == (20,)
    assert np.all(np.isfinite(V))


# ------------------------------------------------------------------ #
# Environment tests — FourRooms (position-only tabular)
# ------------------------------------------------------------------ #

def test_fourrooms_shapes():
    env = FourRoomsTabular(k=2, seed=0)
    S = env.n_states
    assert S > 0
    assert env.P.shape == (4, S, S)   # left, up, right, down
    assert env.n_states == env.n_positions
    np.testing.assert_allclose(env.P.sum(axis=2), np.ones((4, S)), atol=1e-10)


def test_fourrooms_reward_shift():
    env = FourRoomsTabular(k=1, seed=0)
    r2 = env.shift_reward()
    # Gaussian-mixture: all states may be nonzero; check normalization and positivity
    assert r2.sum() > 0
    assert np.all(r2 >= 0)
    assert r2.max() <= 1.0 + 1e-9


def test_fourrooms_transport_modes_modify_kernel():
    symmetric = FourRoomsTabular(k=1, seed=0, one_way_prob=1.0)
    one_way = FourRoomsTabular(k=1, seed=0, one_way_prob=0.05)
    windy = FourRoomsTabular(k=1, seed=0, drift=(0, 1))

    assert symmetric.belt_cycle_states()
    assert not np.allclose(one_way.P, symmetric.P)
    assert not np.allclose(windy.P, symmetric.P)
    assert not np.allclose(one_way.P, windy.P)
    np.testing.assert_allclose(one_way.P.sum(axis=2), np.ones((one_way.n_actions, one_way.n_states)), atol=1e-10)
    np.testing.assert_allclose(windy.P.sum(axis=2), np.ones((windy.n_actions, windy.n_states)), atol=1e-10)


def test_fourrooms_transport_validation():
    with pytest.raises(ValueError):
        FourRoomsTabular(one_way_prob=-0.1)
    with pytest.raises(ValueError):
        FourRoomsTabular(one_way_prob=1.1)
    with pytest.raises(ValueError):
        FourRoomsTabular(drift=(1, 1))
    with pytest.raises(ValueError):
        FourRoomsTabular(drift=(0, 1), drift_strength=1.1)


# ------------------------------------------------------------------ #
# Environment tests — FourRooms (4-action LURD semantics)
# ------------------------------------------------------------------ #

def test_minigrid_shapes():
    env = FourRoomsTabular(k=1, seed=0)
    S, A = env.n_states, env.n_actions
    assert S > 0
    assert A == 4    # left, up, right, down
    assert env.n_states == env.n_positions
    assert env.P.shape == (A, S, S)
    np.testing.assert_allclose(env.P.sum(axis=2), np.ones((A, S)), atol=1e-10)


def test_minigrid_reward_sparsity():
    """Gaussian-mixture reward: nonnegative, normalized to max=1, nonzero sum."""
    env = FourRoomsTabular(k=1, seed=0)
    r = env.r
    assert r.sum() > 0
    assert np.all(r >= 0)
    assert r.max() <= 1.0 + 1e-9


def test_minigrid_shift():
    env = FourRoomsTabular(k=2, seed=0)
    r2 = env.shift_reward()
    assert r2.sum() > 0
    assert np.all(r2 >= 0)
    assert r2.max() <= 1.0 + 1e-9


def test_minigrid_plan():
    env = FourRoomsTabular(k=1, seed=0)
    alg = StandardPS(env.P, env.r, gamma=env.gamma)
    V = alg.plan(K=20)
    assert V.shape == (env.n_states,)
    assert np.all(np.isfinite(V))


# ------------------------------------------------------------------ #
# Model learner tests
# ------------------------------------------------------------------ #

def test_model_learner_stochastic():
    """Model learner should produce row-stochastic P_hat."""
    learner = CountModel(n_states=10, n_actions=3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = int(rng.integers(0, 10))
        a = int(rng.integers(0, 3))
        sp = int(rng.integers(0, 10))
        r = float(rng.uniform(0, 1))
        learner.update(s, a, sp, r)
    P_hat, r_hat = learner.get_model()
    np.testing.assert_allclose(P_hat.sum(axis=2), np.ones((3, 10)), atol=1e-8)
    assert r_hat.shape == (10,)


# ------------------------------------------------------------------ #
# Spectral utility tests
# ------------------------------------------------------------------ #

def test_out_laplacian():
    P = np.array([[0.5, 0.5], [0.3, 0.7]])
    L = out_laplacian(P).toarray()
    expected = np.eye(2) - P
    np.testing.assert_allclose(L, expected, atol=1e-6)


def test_slem_simple():
    n = 10
    P = np.zeros((n, n))
    P[0, 0] = 0.5; P[0, 1] = 0.5
    for i in range(1, n - 1):
        P[i, i - 1] = 0.5; P[i, i + 1] = 0.5
    P[-1, -2] = 0.5; P[-1, -1] = 0.5
    slem, _ = slem_arnoldi(P, k=5)
    assert 0.0 < slem < 1.0


def test_resolvent_forward():
    env = GarnetEnv(n_states=30, n_actions=4, branching=3, seed=2)
    policy = np.zeros(30, dtype=int)
    P_pi = env.P[policy, np.arange(30), :]
    eta = (1 - env.gamma) / env.gamma
    f = resolvent_forward(P_pi, env.r, eta)
    assert f.shape == (30,)
    assert np.all(np.isfinite(f))


# ------------------------------------------------------------------ #
# Algorithm tests
# ------------------------------------------------------------------ #

def test_dynaps_runs():
    env = GarnetEnv(n_states=30, n_actions=4, branching=3, seed=3)
    alg = StandardPS(env.P, env.r, gamma=env.gamma)
    V = alg.plan(K=50)
    assert V.shape == (30,)


def test_gtaps_base_runs():
    env = GarnetEnv(n_states=30, n_actions=4, branching=3, seed=4)
    alg = GtaPS(env.P, env.r, gamma=env.gamma)
    V = alg.plan(K=50)
    assert V.shape == (30,)


# ------------------------------------------------------------------ #
# Metrics tests
# ------------------------------------------------------------------ #

def test_value_error_inf():
    V = np.array([1.0, 2.0, 3.0])
    V_star = np.array([1.0, 2.5, 3.0])
    assert value_error_inf(V, V_star) == pytest.approx(0.5)


def test_steps_to_epsilon():
    V_star = np.array([10.0, 10.0])
    history = [np.array([5.0, 5.0]), np.array([9.5, 9.5]), np.array([10.0, 10.0])]
    k = steps_to_epsilon(history, V_star, eps=0.1)
    assert k == 1

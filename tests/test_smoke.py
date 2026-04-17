"""
Fast smoke tests — verify all imports and basic shapes, < 10s total.
"""
import numpy as np
import pytest

from gta_ps.envs.garnet import GARNETEnv
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms.dyna_ps import DynaPS
from gta_ps.algorithms.gta_ps import GTAPS
from gta_ps.utils.spectral import out_laplacian, in_laplacian, slem_arnoldi, resolvent_forward
from gta_ps.utils.metrics import value_error_inf, steps_to_epsilon


# ------------------------------------------------------------------ #
# Environment tests
# ------------------------------------------------------------------ #

def test_garnet_shapes():
    env = GARNETEnv(n_states=20, n_actions=4, branching=3, seed=0)
    assert env.P.shape == (4, 20, 20)
    assert env.r.shape == (20,)
    # row-stochastic
    np.testing.assert_allclose(env.P.sum(axis=2), np.ones((4, 20)), atol=1e-10)


def test_garnet_vi():
    env = GARNETEnv(n_states=20, n_actions=4, branching=3, seed=1)
    V = env.value_iteration()
    assert V.shape == (20,)
    assert np.all(np.isfinite(V))


def test_fourrooms_shapes():
    env = FourRoomsTabular(k=2, seed=0)
    S = env.n_states
    assert S > 0
    assert env.P.shape == (4, S, S)
    np.testing.assert_allclose(env.P.sum(axis=2), np.ones((4, S)), atol=1e-10)


def test_fourrooms_reward_shift():
    env = FourRoomsTabular(k=1, seed=0)
    r1 = env.r.copy()
    r2 = env.shift_reward()
    # Should have exactly k=1 nonzero entries
    assert np.count_nonzero(r2) == 1


# ------------------------------------------------------------------ #
# Spectral utility tests
# ------------------------------------------------------------------ #

def test_out_laplacian():
    P = np.array([[0.5, 0.5], [0.3, 0.7]])
    L = out_laplacian(P).toarray()
    expected = np.eye(2) - P
    np.testing.assert_allclose(L, expected, atol=1e-12)


def test_slem_simple():
    # Random walk on 10-state chain
    n = 10
    P = np.zeros((n, n))
    P[0, 0] = 0.5; P[0, 1] = 0.5
    for i in range(1, n - 1):
        P[i, i - 1] = 0.5; P[i, i + 1] = 0.5
    P[-1, -2] = 0.5; P[-1, -1] = 0.5
    slem = slem_arnoldi(P, k=5)
    assert 0.0 < slem < 1.0


def test_resolvent_forward():
    env = GARNETEnv(n_states=30, n_actions=4, branching=3, seed=2)
    policy = np.zeros(30, dtype=int)
    P_pi = env.policy_transition(policy)
    eps = (1 - env.gamma) / env.gamma
    f = resolvent_forward(P_pi, env.r, eps)
    assert f.shape == (30,)
    assert np.all(np.isfinite(f))


# ------------------------------------------------------------------ #
# Algorithm tests
# ------------------------------------------------------------------ #

def test_dynaps_runs():
    env = GARNETEnv(n_states=30, n_actions=4, branching=3, seed=3)
    alg = DynaPS(env.P, env.r, gamma=env.gamma)
    V = alg.plan(K=50)
    assert V.shape == (30,)


def test_gtaps_base_runs():
    env = GARNETEnv(n_states=30, n_actions=4, branching=3, seed=4)
    alg = GTAPS(env.P, env.r, gamma=env.gamma)
    V = alg.plan(K=50)
    assert V.shape == (30,)


def test_gtaps_adaptive_runs():
    env = GARNETEnv(n_states=30, n_actions=4, branching=3, seed=5)
    alg = GTAPS(
        env.P, env.r, gamma=env.gamma,
        adaptive_alpha=True, adaptive_beta=True,
    )
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
    assert k == 1   # ||9.5 - 10||_inf = 0.5 <= 0.1*10 = 1.0

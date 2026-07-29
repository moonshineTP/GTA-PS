import numpy as np
import pytest

from gta_ps.algorithms.vast import (
    GtaPsFa,
    LinearStateProjection,
    LinearTabularModel,
    VaST,
)
from gta_ps.envs.fourrooms import FourRoomsTabular


def _self_loop_kernel(n_states: int) -> np.ndarray:
    return np.eye(n_states, dtype=float)[None, :, :]


def test_linear_projection_builds_a_deterministic_table_of_size_a():
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )

    first = LinearStateProjection(a=4).fit_transform(features)
    second = LinearStateProjection(a=4).fit_transform(features)

    np.testing.assert_array_equal(first, second)
    assert np.unique(first).size == 4
    assert first.min() == 0
    assert first.max() == 3


@pytest.mark.parametrize("a", [0, -1])
def test_linear_projection_rejects_nonpositive_abstraction_size(a):
    with pytest.raises(ValueError, match="a must be positive"):
        LinearStateProjection(a=a)


def test_linear_tabular_model_preserves_transition_probability():
    model = LinearTabularModel(
        _self_loop_kernel(4),
        np.arange(4, dtype=float)[:, None],
        a=2,
    )

    np.testing.assert_allclose(model.P.sum(axis=2), 1.0)


@pytest.mark.parametrize("a", [24, 40, 64])
def test_fourrooms_projection_realizes_requested_abstraction_size(a):
    env = FourRoomsTabular(room_size=5, seed=0)

    labels = LinearStateProjection(a=a).fit_transform(env._pos_arr)

    assert np.unique(labels).size == a


def test_vast_runs_vi_on_the_compact_table_and_lifts_values():
    features = np.arange(4, dtype=float)[:, None]
    planner = VaST(
        _self_loop_kernel(4),
        np.array([1.0, 1.0, 3.0, 3.0]),
        features,
        gamma=0.5,
        a=2,
        batch_size=1,
    )

    value = planner.plan(K=100)

    assert planner.n_abstract_states == 2
    np.testing.assert_allclose(value, np.array([2.0, 2.0, 6.0, 6.0]), atol=1e-8)


def test_vast_reward_update_preserves_the_compact_transition_model():
    features = np.arange(4, dtype=float)[:, None]
    planner = VaST(
        _self_loop_kernel(4),
        np.zeros(4),
        features,
        gamma=0.5,
        a=2,
        batch_size=1,
    )
    planner.warm_start(np.zeros(4))

    value = planner.plan(K=100, r=np.array([2.0, 2.0, 4.0, 4.0]))

    np.testing.assert_allclose(value, np.array([4.0, 4.0, 8.0, 8.0]), atol=1e-8)


def test_gta_ps_fa_runs_on_the_same_compact_table():
    features = np.arange(4, dtype=float)[:, None]
    planner = GtaPsFa(
        _self_loop_kernel(4),
        np.array([1.0, 1.0, 3.0, 3.0]),
        features,
        gamma=0.5,
        a=2,
        batch_size=1,
        q=None,
    )
    planner.warm_start(np.zeros(4))

    value = planner.plan(K=100)

    assert planner.n_abstract_states == 2
    np.testing.assert_allclose(value, np.array([2.0, 2.0, 6.0, 6.0]), atol=2e-4)


def test_vast_batch_size_freezes_the_pre_backup_residual_order():
    kernel = np.zeros((1, 3, 3), dtype=float)
    kernel[0, 0, 2] = 1.0
    kernel[0, 1, 0] = 1.0
    kernel[0, 2, 2] = 1.0
    features = np.arange(3, dtype=float)[:, None]
    rewards = np.array([3.0, 0.0, 1.0])

    eager = VaST(
        kernel,
        rewards,
        features,
        gamma=0.9,
        a=3,
        batch_size=1,
    )
    frozen = VaST(
        kernel,
        rewards,
        features,
        gamma=0.9,
        a=3,
        batch_size=2,
    )

    eager_value = eager.plan(K=2)
    frozen_value = frozen.plan(K=2)

    assert eager_value[1] == pytest.approx(2.7)
    assert frozen_value[1] == pytest.approx(0.0)
    assert frozen_value[2] == pytest.approx(1.0)

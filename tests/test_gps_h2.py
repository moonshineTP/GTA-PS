import numpy as np

from gta_ps.algorithms import DpHost
from gta_ps.algorithms.batching import BatchedPriorityModule
from gta_ps.algorithms.gps_h2 import GpsH2
from gta_ps.algorithms.standard_ps import StandardPS
from gta_ps.envs.fourrooms import FourRoomsTabular


def _chain_kernel(n_states: int = 3) -> np.ndarray:
    p = np.zeros((1, n_states, n_states), dtype=float)
    for state in range(n_states):
        p[0, state, state] = 0.5
        p[0, state, (state + 1) % n_states] = 0.5
    return p


def test_gps_h2_normalizes_value_term():
    algorithm = GpsH2(
        _chain_kernel(),
        np.ones(3),
        gamma=0.95,
        batch_size=2,
        upsilon=0.05,
    )
    algorithm.xi = np.array([0.20, 0.10, 0.30])
    algorithm.V = np.array([10.0, 0.0, 1.0])

    np.testing.assert_allclose(
        algorithm.priority_scores(),
        np.array([0.70, 0.10, 0.35]),
    )

    algorithm._rebuild_heap()
    np.testing.assert_allclose(
        algorithm.priority_scores(),
        np.array([4.75, 1.475, 5.275]),
    )


def test_gps_h2_selects_a_fixed_block_instead_of_top_m_states():
    kernel = np.eye(4, dtype=float)[None, :, :]
    algorithm = GpsH2(
        kernel,
        np.array([0.90, 0.05, 1.00, 0.10]),
        gamma=0.95,
        batch_size=2,
        blocks=((0, 1), (2, 3)),
    )

    assert algorithm.pop_and_prioritize() == 3
    algorithm.update_value(3, 100.0)
    assert algorithm.pop_and_prioritize() == 2


def test_gps_h2_initializes_modified_topological_order_once():
    kernel = np.zeros((1, 3, 3), dtype=float)
    kernel[0, 0, 1] = 1.0
    kernel[0, 1, 2] = 1.0
    kernel[0, 2, 2] = 1.0

    algorithm = GpsH2(
        kernel,
        np.ones(3),
        gamma=0.95,
        batch_size=3,
    )

    assert algorithm.block_orders == ((2, 1, 0),)


def test_gps_h2_accepts_nonnegative_fourrooms_reward_resamples():
    env = FourRoomsTabular(room_size=5, k=2, seed=0)
    agent = DpHost.from_arrays(
        env.P,
        env.r,
        0.95,
        GpsH2(env.P, env.r, gamma=0.95, batch_size=4),
    )
    agent.V = env.value_iteration(env.r)
    agent.module.V = agent.V.copy()

    agent.plan(K=20, r=env.shift_reward())
    agent.plan(K=20, r=env.shift_reward())


def test_batched_module_freezes_h1_order_until_the_batch_is_consumed():
    kernel = np.eye(3, dtype=float)[None, :, :]
    inner = StandardPS(kernel, np.array([1.0, 2.0, 3.0]), gamma=0.95)
    batched = BatchedPriorityModule(inner, batch_size=2)

    assert batched.pop_and_prioritize() == 2
    batched.update_value(2, 100.0)
    assert batched.pop_and_prioritize() == 1


class _ManualBatchModule:
    def __init__(self, batch: list[int]) -> None:
        self.batch = list(batch)
        self.V = np.zeros(2)

    def reward_update(self, delta_r: np.ndarray) -> None:
        return None

    def pop_batch(self, limit: int) -> list[int]:
        if limit <= 0 or not self.batch:
            return []
        out = self.batch[:limit]
        self.batch = self.batch[limit:]
        return out

    def pop_and_prioritize(self) -> int | None:
        return self.batch.pop(0) if self.batch else None

    def push_predecessors(self, s: int, xi_s: float) -> None:
        return None

    def notify_backup(self, s: int, adv_PS_row: np.ndarray, xi_s: float) -> None:
        return None

    def update_model(self, P: np.ndarray, r: np.ndarray) -> None:
        return None

    def update_value(self, s: int, v_s: float) -> None:
        self.V[s] = v_s

    def reset(self) -> None:
        self.batch = []
        self.V[:] = 0.0


def test_dp_host_applies_batched_priorities_simultaneously():
    kernel = np.zeros((1, 2, 2), dtype=float)
    kernel[0, 0, 1] = 1.0
    kernel[0, 1, 1] = 1.0
    module = _ManualBatchModule([1, 0])
    agent = DpHost.from_arrays(kernel, np.array([0.0, 1.0]), 0.9, module)

    value = agent.plan(K=2)

    np.testing.assert_allclose(value, np.array([0.0, 1.0]))

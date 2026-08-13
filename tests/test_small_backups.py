import numpy as np
import pytest

from gta_ps.algorithms import make_agent
from gta_ps.algorithms.empirical_model import SampleModel


def test_sample_model_tracks_sparse_transition_frequencies() -> None:
    model = SampleModel(n_states=3, n_actions=1)

    model.update(0, 0, 1, 0.0)
    model.update(0, 0, 2, 0.0)
    model.update(0, 0, 2, 0.0)
    model.update(0, 0, 2, 0.0)

    assert model.get_row(0, 0) == pytest.approx([0.0, 0.25, 0.75])
    assert model.predecessors(2) == [(0, 0, 0.75)]


def test_sample_model_dyna_real_step_uses_running_small_backup() -> None:
    P = np.ones((1, 1, 1), dtype=float)
    r = np.zeros(1, dtype=float)
    agent = make_agent(
        "standard_ps",
        host="dyna",
        P=P,
        r=r,
        gamma=0.9,
        eta_td=0.01,
        model_type="sample",
    )

    agent.real_step(0, 0, 2.0, 0)
    agent.real_step(0, 0, 4.0, 0)

    assert agent.Q[0, 0] == pytest.approx(3.0)


def test_sample_model_dyna_planning_propagates_delta_to_counted_predecessors() -> None:
    P = np.zeros((1, 3, 3), dtype=float)
    P[0, 0, 1] = 1.0
    P[0, 1, 2] = 1.0
    P[0, 2, 2] = 1.0
    r = np.zeros(3, dtype=float)
    agent = make_agent(
        "standard_ps",
        host="dyna",
        P=P,
        r=r,
        gamma=0.9,
        eta_td=0.01,
        model_type="sample",
    )

    agent.real_step(0, 0, 0.0, 1)
    agent.real_step(1, 0, 1.0, 2)
    agent.plan_steps(1)

    assert agent.module.V[1] == pytest.approx(1.0)
    assert agent.Q[0, 0] == pytest.approx(0.9)
    assert agent.completed_backups == 1

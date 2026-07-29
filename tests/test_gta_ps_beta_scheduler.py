import numpy as np
import pytest

from gta_ps.algorithms.gta_ps import GtaPS


def _chain_kernel(n_states: int = 4) -> np.ndarray:
    p = np.zeros((1, n_states, n_states), dtype=float)
    for s in range(n_states):
        p[0, s, s] = 0.5
        p[0, s, (s + 1) % n_states] = 0.5
    return p


def test_adaptive_beta_uses_additive_offset():
    alg = GtaPS(
        _chain_kernel(),
        np.ones(4),
        adaptive_beta=True,
        beta1=2.0,
        beta2=0.25,
    )

    np.testing.assert_allclose(alg._beta_t(0.9), 0.7)
    np.testing.assert_allclose(alg._beta_t(0.5), 1.5)


def test_negative_beta_preserves_standard_ps_residual_priority():
    alg = GtaPS(
        _chain_kernel(),
        np.ones(4),
        adaptive_beta=False,
        beta=-2.0,
    )
    alg.xi = np.array([0.2, 0.1, 0.0, 0.0])
    alg.tau = np.array([10.0, 0.01, 0.0, 0.0])
    alg._top_q = set(range(4))

    alg._rekey_residual_queue()

    queued = {s: -neg_priority for neg_priority, _version, s in alg._Q_xi}
    assert set(queued) == {0, 1}
    assert queued[0] == 0.2
    assert queued[1] == 0.1
    assert alg.pop_and_prioritize() == 0


def test_tau_refresh_instrumentation_tracks_count_and_backup_age():
    alg = GtaPS(_chain_kernel(), np.ones(4))
    initial_count = alg._tau_refresh_count

    alg._step_count = 7
    alg._refresh_tau()

    assert alg._tau_refresh_count == initial_count + 1
    assert alg._last_tau_refresh_step == 7


def test_explicit_slem_recompute_period_overrides_sqrt_default():
    alg = GtaPS(
        _chain_kernel(),
        np.ones(4),
        slem_recompute_every=17,
    )

    assert alg._T_arn == 17


def test_slem_refresh_instrumentation_tracks_initial_and_manual_refreshes():
    alg = GtaPS(_chain_kernel(), np.ones(4))

    assert alg._slem_refresh_count == 1
    alg._refresh_slem()
    assert alg._slem_refresh_count == 2


def test_slem_recompute_period_must_be_positive():
    with pytest.raises(ValueError, match="slem_recompute_every"):
        GtaPS(
            _chain_kernel(),
            np.ones(4),
            slem_recompute_every=0,
        )

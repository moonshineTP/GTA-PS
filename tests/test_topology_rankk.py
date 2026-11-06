import numpy as np
import scipy.sparse as sp
import pytest

from gta_ps.algorithms.gta_ps import GtaPS
from gta_ps.utils.spectral import in_laplacian


def _kernel() -> np.ndarray:
    kernel = np.zeros((2, 4, 4), dtype=float)
    kernel[0] = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    kernel[1] = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    return kernel


def _exact_resolvents(module: GtaPS) -> tuple[np.ndarray, np.ndarray]:
    n = module.n_states
    identity = sp.eye(n, format="csr")
    A_out = identity * (1.0 + module.eta) - sp.csr_array(module.P_pi)
    A_in = in_laplacian(module.P_pi) + identity * module.eta
    return (
        np.linalg.solve(A_out.toarray(), module.r),
        np.linalg.solve(A_in.toarray(), module.r),
    )


def test_topology_update_lazy_matches_exact_rank_k_resolvent_without_lu_rebuild(monkeypatch: pytest.MonkeyPatch):
    module = GtaPS(_kernel(), np.array([0.2, 0.4, 0.6, 0.8]), gamma=0.95, q=None)

    updates = {
        0: np.array([0.25, -0.25]),
        2: np.array([-0.20, 0.20]),
    }
    for state, delta_pi in updates.items():
        module.pi_PS[state] += delta_pi
        module.P_pi[state] = (module.pi_PS[state, :, None] * module.P[:, state, :]).sum(axis=0)
        module._pending_delta_pi[state] += delta_pi

    module._has_pending_topo = True
    f_out_exact, f_in_exact = _exact_resolvents(module)

    def _unexpected_rebuild(*args, **kwargs):
        raise AssertionError("rank-k update should not rebuild LU for small systems")

    monkeypatch.setattr("gta_ps.algorithms.gta_ps.spla.splu", _unexpected_rebuild)

    module.topology_update_lazy()

    np.testing.assert_allclose(module.f_out, f_out_exact, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(module.f_in, f_in_exact, rtol=1e-9, atol=1e-9)

    delta_pi = np.array([0.10, -0.10])
    module.pi_PS[1] += delta_pi
    module.P_pi[1] = (module.pi_PS[1, :, None] * module.P[:, 1, :]).sum(axis=0)
    module._pending_delta_pi[1] += delta_pi
    module._has_pending_topo = True
    f_out_exact, f_in_exact = _exact_resolvents(module)

    module.topology_update_lazy()

    np.testing.assert_allclose(module.f_out, f_out_exact, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(module.f_in, f_in_exact, rtol=1e-9, atol=1e-9)


def test_solver_threshold_zero_forces_gmres():
    module = GtaPS(
        _kernel(),
        np.array([0.2, 0.4, 0.6, 0.8]),
        gamma=0.95,
        q=None,
        solver_threshold=0,
    )

    assert module._lu_out is None
    assert module._lu_in is None

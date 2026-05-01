"""
Assertions that catch K unit confusion (per-step vs per-shift vs total-cap).

All tests use tiny MDPs so they run in < 1 s.
"""
import sys
import inspect
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.algorithms.standard_ps import StandardPS
from gta_ps.algorithms.gta_ps import GtaPS


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _tiny(n: int = 6, a: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(n), size=(a, n))   # (a, n, n) after reshape
    P = raw.reshape(a, n, n)
    P /= P.sum(axis=2, keepdims=True)
    r = rng.random(n)
    return P, r


def _make_alg(cls, P, r):
    """Construct with arnoldi_k=2 so Arnoldi succeeds on tiny MDPs (n=6)."""
    if cls is GtaPS:
        return cls(P, r, gamma=0.95, arnoldi_k=2)
    return cls(P, r, gamma=0.95)


# ---------------------------------------------------------------------------
# 1. plan(K) consumes at most K backups per call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [StandardPS, GtaPS])
def test_plan_consumes_at_most_K_backups(cls):
    """plan(K) must pop at most K states — K is a per-call budget, not a hint."""
    P, r = _tiny()
    alg = _make_alg(cls, P, r)

    popped: list = []
    _orig = alg.pop_and_prioritize
    def _counting():
        s = _orig()
        if s is not None:
            popped.append(s)
        return s
    alg.pop_and_prioritize = _counting  # type: ignore[method-assign]

    K = 3
    alg.plan(K=K)
    assert len(popped) <= K, (
        f"{cls.__name__}.plan(K={K}) popped {len(popped)} states — "
        "K is leaking across host steps"
    )


# ---------------------------------------------------------------------------
# 2. Each plan() call gets an independent budget of K
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [StandardPS, GtaPS])
def test_second_plan_call_gets_fresh_K_budget(cls):
    """A second plan(K) call must also pop at most K states — not K − (already done)."""
    P, r = _tiny()
    alg = _make_alg(cls, P, r)

    popped: list = []
    _orig = alg.pop_and_prioritize
    def _counting():
        s = _orig()
        if s is not None:
            popped.append(s)
        return s
    alg.pop_and_prioritize = _counting  # type: ignore[method-assign]

    K = 3
    alg.plan(K=K)
    first_count = len(popped)

    # Reset counter; run second independent host step
    popped.clear()
    alg.plan(K=K)
    second_count = len(popped)

    assert second_count <= K, (
        f"{cls.__name__}: second plan(K={K}) popped {second_count} > K — "
        "budget is not resetting between host steps"
    )
    # Combined must be ≤ 2K (each step is independent)
    assert first_count + second_count <= 2 * K


# ---------------------------------------------------------------------------
# 3. Total backups across n_shifts must be ≤ K * n_shifts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [StandardPS, GtaPS])
def test_total_backups_bounded_by_K_times_n_shifts(cls):
    """Running n_shifts host steps must total at most K * n_shifts backups."""
    P, r = _tiny()
    alg = _make_alg(cls, P, r)

    total_popped = 0
    _orig = alg.pop_and_prioritize
    def _counting():
        nonlocal total_popped
        s = _orig()
        if s is not None:
            total_popped += 1
        return s
    alg.pop_and_prioritize = _counting  # type: ignore[method-assign]

    K = 2
    n_shifts = 5
    rng = np.random.default_rng(1)
    n = P.shape[1]
    for _ in range(n_shifts):
        r_new = rng.random(n)
        alg.plan(K=K, r=r_new)

    assert total_popped <= K * n_shifts, (
        f"Total backups {total_popped} > K*n_shifts={K * n_shifts} — "
        "plan() is treating K as a total-shift budget, not a per-step budget"
    )


# ---------------------------------------------------------------------------
# 4. convergence_benchmark uses distinct names for per-step vs total-cap
# ---------------------------------------------------------------------------

def test_convergence_benchmark_uses_distinct_K_names():
    """steps_to_converge must have 'K' (per-step) and 'budget_cap' (total) as separate params."""
    import convergence_benchmark as cb  # misc/ is on sys.path via ROOT insert above

    sig = inspect.signature(cb.steps_to_converge)
    params = list(sig.parameters.keys())

    assert "K" in params, (
        "steps_to_converge missing per-step param 'K' — "
        "was it renamed back to K_check?"
    )
    assert "budget_cap" in params, (
        "steps_to_converge missing total-cap param 'budget_cap' — "
        "was it renamed back to K_max?"
    )
    assert "K" != "budget_cap"  # tautology but documents the invariant

    # run_convergence must forward both
    sig2 = inspect.signature(cb.run_convergence)
    params2 = list(sig2.parameters.keys())
    assert "K" in params2, "run_convergence must expose per-step 'K'"
    assert "budget_cap" in params2, "run_convergence must expose 'budget_cap'"

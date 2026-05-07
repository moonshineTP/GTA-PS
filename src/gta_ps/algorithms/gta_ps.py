"""
GTA-PS Computation Management Module (memo §Algorithm and Complexity).

Exposes named procedures: GtaPolicyUpdate, TopologyUpdate, RewardUpdate,
PopAndPrioritize, PushPredecessors, RefreshTopQ, RekeyResidualQueue.

Hosts (DpHost, DynaHost) call these and own the Bellman/TD update.
The topology policy π^PS and LU factorizations are always derived from the
ORACLE P. The empirical _P_model is used only for predecessor lookup and
value-residual computation in push_predecessors (synced by DynaHost).
"""
from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import softmax as _sp_softmax

from .base import PrioritizedSweepingModule, BellmanMixin, TdMixin
from ..utils.spectral import slem_arnoldi, in_laplacian

# LU + Sherman-Morrison column cache for n ≤ threshold; GMRES above.
_LU_THRESHOLD = 300


class GtaPS(PrioritizedSweepingModule, BellmanMixin):
    """
    GTA-PS computation management module.

    Parameters
    ----------
    P : ndarray (|A|, |S|, |S|)
    r : ndarray (|S|,)
    gamma : float = 0.95
     alpha : float = 0.5                 fixed forward→backward mix
    beta  : float = 1.0                 base topology weight (adaptive_beta=False)
    beta1 : float = 1.0                 schedule amplitude
    beta2 : float = 0.1                 schedule exponent / offset
    beta_mode : str = "linear"          schedule mode: "fixed", "linear", "log"
    adaptive_beta  : bool = False       
    eta   : float | None = None         resolvent ε; None → ε* = (1−γ)/γ
    b     : float = 1.0                 Boltzmann temperature for π^PS
    arnoldi_k : int = 15
    q     : int | None = None           frontier size; None → |S|
    """

    def __init__(
        self,
        P: np.ndarray,
        r: np.ndarray,
        gamma: float = 0.95,
        alpha: float = 0.5,
        beta: float = 1.0,
        beta1: float = 1.0,
        beta2: float = 0.1,
        beta_mode: str = "linear",
        adaptive_beta: bool = False,
        eta: Optional[float] = None,
        b: float = 1.0,
        arnoldi_k: int = 15,
        q: Optional[int] = None,
        # legacy kwarg — accepted but ignored (schedule was always linear in old code)
        slem_recompute_every: Optional[int] = None,
    ):
        self.P = P                  # oracle — topology always uses this
        self._P_model = P.copy()   # empirical; DynaHost replaces with P_hat
        self._r_model = r.copy()
        self.r = r.copy()

        self.gamma = gamma
        self.n_states = P.shape[1]
        self.n_actions = P.shape[0]

        self.alpha_base = alpha
        self.beta_base = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.adaptive_beta = adaptive_beta
        self.eta = eta if eta is not None else (1.0 - gamma) / gamma
        self.b = b
        self.arnoldi_k = arnoldi_k
        self.q = q
        self.beta_mode = beta_mode

        self._init_topology()



    # ------------------------------------------------------------------
    # Linear-system helpers (auto-select LU vs GMRES based on n)
    # ------------------------------------------------------------------

    def _solve_out(self, rhs: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        if self._lu_out is not None:
            return self._lu_out.solve(rhs)
        f, info = spla.gmres(self._A_out, rhs, x0=x0, restart=30, maxiter=500, atol=1e-4)
        if info != 0:
            f = spla.spsolve(self._A_out.tocsc(), rhs)
        return f

    def _solve_in(self, rhs: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        if self._lu_in is not None:
            return self._lu_in.solve(rhs)
        f, info = spla.gmres(self._A_in, rhs, x0=x0, restart=30, maxiter=500, atol=1e-4)
        if info != 0:
            f = spla.spsolve(self._A_in.tocsc(), rhs)
        return f

    def _rebuild_sparse_systems(self) -> None:
        """Rebuild _A_out/_A_in from current P_pi (large-n path, after policy change)."""
        n, eta = self.n_states, self.eta
        Psp = sp.csr_array(self.P_pi)
        Psp.data[np.abs(Psp.data) < 1e-6] = 0.0
        Psp.eliminate_zeros()
        self._A_out = sp.eye(n, format="csr") * (1.0 + eta) - Psp
        self._A_in = in_laplacian(self.P_pi) + sp.eye(n, format="csr") * eta



    # ------------------------------------------------------------------
    # Initialisation / precompute (memo Algorithm lines 1-7, oracle P)
    # ------------------------------------------------------------------

    def _init_topology(self) -> None:
        n = self.n_states
        eta = self.eta
        use_lu = n <= _LU_THRESHOLD

        self.V = np.zeros(n)
        self.xi = np.zeros(n)
        self.tau = np.zeros(n)

        # Boltzmann topology policy π^PS from oracle P (V=0 initially)
        Q_0 = self.r[None, :] + self.gamma * (
            self.P * self.V[None, None, :]
        ).sum(axis=2)  # (|A|, |S|)
        pi_all = _sp_softmax(self.b * Q_0, axis=0)   # (|A|, |S|) softmax over actions
        self.pi_PS = pi_all.T.copy()                   # (|S|, |A|)
        self.P_pi = np.einsum("sa,ast->st", self.pi_PS, self.P)  # (|S|, |S|)

        # Always store sparse system matrices (needed for large-n GMRES path and rebuilds)
        Psp = sp.csr_array(self.P_pi)
        Psp.data[np.abs(Psp.data) < 1e-6] = 0.0
        Psp.eliminate_zeros()
        self._A_out = sp.eye(n, format="csr") * (1.0 + eta) - Psp
        self._A_in = in_laplacian(self.P_pi) + sp.eye(n, format="csr") * (eta + 1e-6)

        if use_lu:
            self._lu_out = spla.splu(self._A_out.tocsc())
            self._lu_in = spla.splu(self._A_in.tocsc())
        else:
            self._lu_out = None
            self._lu_in = None

        # Resolvent fields
        self.f_out = self._solve_out(self.r)
        self.f_in = self._solve_in(self.r)

        # SLEM and schedules
        self.slem = 0.0
        self.beta_t = self._beta_t(self.slem)
        self.alpha_t = self._alpha_t(self.slem)
        self._v0_arnoldi = None
        self._refresh_slem()

        # Priority queues and counters
        self.q_t = self.q if self.q is not None else n
        self._step_count = 0.
        self._T_arn = max(1, math.isqrt(max(1, n) - 1) + 1)
        self._T_topo = max(1, n)
        self._T_tau = max(1, n)
        self._tau_refresh_count = 0
        self._pending_delta_pi = np.zeros((n, self.n_actions))
        self._has_pending_topo = False
        self._version_xi: dict = {}
        self._top_q: set = set()
        self._H_q: list = []
        self._version_hq: dict = {}

        # Initialise xi from residuals with V=0
        EV_all = (self._P_model * self.V[None, None, :]).sum(axis=2)
        self.xi = np.abs(
            (self._r_model[None, :] + self.gamma * EV_all).max(axis=0) - self.V
        )
        self._refresh_tau()
        self._predecessor_sets = [set() for _ in range(n)]
        self._build_predecessor_cache()


    def _build_predecessor_cache(self) -> None:
        """Precompute predecessors with thresholding."""
        n = self.n_states
        cache: list[list[tuple[int,int]]] = [[] for _ in range(n)]
        for a in range(self.n_actions):
            rows, cols = np.where(self._P_model[a] > 1e-2)
            for pred, succ in zip(rows.tolist(), cols.tolist()):
                if (pred, a) not in self._predecessor_sets[succ]:
                    self._predecessor_sets[succ].add((pred, a))
                    cache[succ].append((pred, a))
        self._predecessors = cache


    def _refresh_slem(self) -> None:
        """Recompute SLEM and the derived beta/alpha schedules."""
        try:
            if self.n_states < 3:
                self.slem = 0.0
            else:
                k_arn = min(self.arnoldi_k, self.n_states - 2)
                self.slem, self._v0_arnoldi = slem_arnoldi(
                    self.P_pi,
                    k=k_arn,
                    v0=self._v0_arnoldi,
                )
        except Exception:
            return
        self.beta_t = self._beta_t(self.slem)
        self.alpha_t = self._alpha_t(self.slem)


    def update_model_transition(self, s: int, a: int, p_row: np.ndarray, r_s: float) -> None:
        """Incremental model update with O(S) complexity."""
        self._P_model[a, s] = p_row
        self._r_model[s] = r_s
        
        # Incremental predecessor tracking: only sp where P(s, a, sp) > threshold
        for sp_idx in np.where(p_row > 1e-2)[0]:
            if (s, a) not in self._predecessor_sets[sp_idx]:
                self._predecessor_sets[sp_idx].add((s, a))
                self._predecessors[sp_idx].append((s, a))



    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def _beta_t(self, slem: float) -> float:
        if not self.adaptive_beta:
            return self.beta_base
        if self.beta_mode == "fixed":
            return self.beta1
        if self.beta_mode == "linear":
            # β_t = β₁ · ((1−SLEM) + β₂)
            # β₂ is a plain additive offset in spectral-gap space.
            # β₂ < 0  → dampening threshold (dampen; β_t=0 when |β₂| > 1−SLEM)
            # β₂ = 0  → pure spectral-gap scaling (β_t > 0 iff SLEM < 1)
            # β₂ > 0  → positive floor (β_t always > 0 when β₁ > 0)
            return self.beta1 * ((1.0 - slem) + self.beta2)
        if self.beta_mode == "log":
            # β_t = β₁ · (−log SLEM + β₂)
            # −log(SLEM) ≈ (1−SLEM) for SLEM≈1 but saturates as SLEM→0,
            # preventing topology overdrive on fast-mixing chains.
            return self.beta1 * (-np.log(max(slem, 1e-6)) + self.beta2)
        raise ValueError(f"Invalid beta_mode: {self.beta_mode!r}. "
                         f"Choose from: 'fixed', 'linear', 'log'.")

    def _alpha_t(self, slem: float) -> float:
        return self.alpha_base



    # ------------------------------------------------------------------
    # Queue maintenance
    # ------------------------------------------------------------------

    def _rebuild_q_tau(self) -> None:
        n = len(self.tau)
        q = min(self.q_t, n)
        # argpartition is O(n); argsort would be O(n log n) — memo §Complexity claims O(|S|+q log|S|).
        idx = np.argpartition(self.tau, n - q)[n - q:]
        self._top_q = set(idx.tolist())

    def _refresh_top_q(self) -> None:
        """Rebuild top-q set and H_q frontier heap.

        H_q holds states in top-q that have ξ=0 (pure topology candidates).
        Score = β_t·τ[s]; excluded when β_t ≤ 0 (no topology amplification).
        Called only at init and reward shifts — NOT per backup — to avoid
        O(n log n) argsort overhead inside the planning loop.
        """
        self._rebuild_q_tau()
        # Build H_q: top-q states with ξ=0 and positive β_t·τ score
        self._version_hq = {}
        entries = []
        if self.beta_t > 0:
            for s in self._top_q:
                if self.xi[s] <= 1e-4:
                    score = self.beta_t * self.tau[s]
                    if score > 0:
                        self._version_hq[int(s)] = 0
                        entries.append((-score, 0, int(s)))
        heapq.heapify(entries)
        self._H_q = entries


    def _rekey_residual_queue(self) -> None:
        # Gate on ξ[s] > 0 (per memo §Algorithm: Q_ξ = {s : ξ[s] > 0}).
        # Score = ξ + β_t·τ.
        #   β_t > 0 → topology amplifies priority via magnitude of discounted reachability.
        #   β_t < 0 → topology dampens priority (conservative on fast-mixing).
        #   β_t = 0 → pure Standard-PS (no topology signal).
        # Floor at 0 (not at ξ): a residual state must stay in the queue
        # (positive priority) but its relative rank can still be damped when β_t < 0.
        hot = np.where(self.xi > 1e-4)[0]
        self._version_xi = {int(s): 0 for s in hot}
        entries = [
            (-float(self.xi[s] + self.beta_t * self.tau[s]), 0, int(s))
            for s in hot
        ]
        heapq.heapify(entries)
        self._Q_xi = entries

        # Topology frontier H_q: top-q states with xi=0, tau>0 (only when β_t > 0).
        # These are backed up to propagate topology signal even with zero residual.
        topo_entries = []
        self._version_hq = {}
        if self.beta_t > 0:
            for s in self._top_q:
                s = int(s)
                if self.xi[s] <= 1e-4 and self.tau[s] > 0:
                    self._version_hq[s] = 0
                    topo_entries.append((-self.beta_t * float(self.tau[s]), 0, s))
        heapq.heapify(topo_entries)
        self._H_q = topo_entries



    # ------------------------------------------------------------------
    # GtaPolicyUpdate
    # ------------------------------------------------------------------

    def gta_policy_update(
        self, s: int, adv_PS_row: np.ndarray
    ) -> np.ndarray:
        """Update π^PS[s] via softmax on advantage; return Δπ row."""
        # Clip to prevent softmax overflow
        adv_clipped = np.clip(self.b * adv_PS_row, -50, 50)
        new_pi = _sp_softmax(adv_clipped)       # (|A|,)
        delta_pi = new_pi - self.pi_PS[s]
        self.pi_PS[s] += delta_pi
        # Update P_pi row s using oracle P (topology uses oracle graph)
        self.P_pi[s] = (self.pi_PS[s, :, None] * self.P[:, s, :]).sum(axis=0)
        return delta_pi



    def _refresh_tau(self) -> None:
        """Compute a magnitude-based topology field τ and rebuild all queues.

        ξ is already an absolute Bellman residual. Each directed resolvent
        response is also converted to magnitude before mixing, so τ reflects
        propagation strength rather than signed cancellation between the
        out- and in-directed fields.
        """
        tau_out = np.abs(self._solve_out(self.xi))
        tau_in  = np.abs(self._solve_in(self.xi))
        self.tau = (1.0 - self.alpha_t) * tau_out + self.alpha_t * tau_in
        self._rebuild_q_tau()
        self._refresh_top_q()
        self._rekey_residual_queue()


    # ------------------------------------------------------------------
    # TopologyUpdate (Sherman-Morrison on A_out column cache)
    # ------------------------------------------------------------------

    def topology_update_lazy(self) -> None:
        """Apply all accumulated delta_pi changes at once by rebuilding LU."""
        if not self._has_pending_topo:
            return
            
        n = self.n_states
        eta = self.eta
        
        # Rebuild A_out and A_in from the current P_pi (which was updated in gta_policy_update)
        Psp = sp.csr_array(self.P_pi)
        Psp.data[np.abs(Psp.data) < 1e-6] = 0.0
        Psp.eliminate_zeros()
        
        self._A_out = sp.eye(n, format="csr") * (1.0 + eta) - Psp
        self._A_in = in_laplacian(self.P_pi) + sp.eye(n, format="csr") * (eta + 1e-6)
        
        if self._lu_out is not None:
            self._lu_out = spla.splu(self._A_out.tocsc())
            self._lu_in = spla.splu(self._A_in.tocsc())
            self.f_out = self._lu_out.solve(self.r)
            self.f_in = self._lu_in.solve(self.r)
        else:
            self.f_out = self._solve_out(self.r, x0=self.f_out)
            self.f_in = self._solve_in(self.r, x0=self.f_in)

        # Recompute τ = G·ξ with fresh LU factors and current Bellman residuals.
        self._refresh_tau()

        self._pending_delta_pi.fill(0)
        self._has_pending_topo = False

    def topology_update(self, s_upd: int, delta_pi_row: np.ndarray) -> None:
        """Deprecated: use notify_backup which handles lazy accumulation."""
        pass



    # ------------------------------------------------------------------
    # PrioritizedSweepingModule interface
    # ------------------------------------------------------------------

    def reward_update(self, delta_r: np.ndarray) -> None:
        self.r += delta_r
        self._r_model += delta_r
        # Do NOT reset _step_count here: Arnoldi fires on its natural schedule
        # (every _T_arn backup steps). Forcing it on the first post-shift step
        # costs O(n·deg·arnoldi_k) redundantly before any V has changed.
        # A_out/A_in are unchanged by a reward shift — only rhs changes.
        if self._lu_out is not None:
            # Small n: use LU tri-solve (O(n)) instead of g @ delta_r (O(n²)).
            self.f_out += self._lu_out.solve(delta_r)
            self.f_in  += self._lu_in.solve(delta_r)
        else:
            # Large n: GMRES re-solve with updated rhs.
            self.f_out = self._solve_out(self.r)
            self.f_in = self._solve_in(self.r)

        # Recompute xi from current V and the new reward vector.
        # Without this, Q_xi tracks stale residuals from the pre-shift reward;
        # on sparse-reward envs (e.g. FourRooms, 1 non-zero per 104 states) this
        # leaves Q_xi pointing at the wrong goal state for every subsequent shift.
        EV_all = (self._P_model * self.V[None, None, :]).sum(axis=2)
        self.xi = np.abs(
            (self._r_model[None, :] + self.gamma * EV_all).max(axis=0) - self.V
        )

        # τ = G·ξ: full refresh with new xi (done per reward shift, not per backup).
        self._refresh_tau()

    def pop_and_prioritize(self) -> int | None:
        # Peek valid head from Q_xi (residual queue).
        xi_pri, xi_s = None, None
        while self._Q_xi:
            neg_pri, v, s = self._Q_xi[0]
            if self._version_xi.get(s, -1) == v:
                xi_pri, xi_s = -neg_pri, s
                break
            heapq.heappop(self._Q_xi)

        # Peek valid head from H_q (topology frontier).
        hq_pri, hq_s = None, None
        while self._H_q:
            neg_pri, v, s = self._H_q[0]
            if self._version_hq.get(s, -1) == v:
                hq_pri, hq_s = -neg_pri, s
                break
            heapq.heappop(self._H_q)

        if xi_s is None and hq_s is None:
            return None

        # BUDGET PRIORITY:
        # 1. Residual-reducing backups (Q_xi) are MANDATORY for convergence.
        # 2. Topology backups (H_q) are EXPLORATORY (policy updates with xi=0).
        # If we have both, we must not let H_q starve Q_xi, otherwise we
        # perform no-op Bellman backups and error stays flat.
        
        # POP LOGIC: Always exhaust Q_xi first if budget is tight.
        # We only take from H_q if it is significantly higher priority OR Q_xi is empty.
        # Given the derailed lines, we enforce: Residuals first.
        if xi_s is not None:
            heapq.heappop(self._Q_xi)
            chosen = xi_s
        else:
            heapq.heappop(self._H_q)
            chosen = hq_s

        self._last_beta = self.beta_t
        self._last_signal_ratio = float(
            np.sum(np.abs(self.beta_t * self.tau))
            / max(float(np.sum(self.xi)), 1e-6)
        )
        self._slem = self.slem
        return chosen

    def push_predecessors(self, s: int, xi_s: float) -> None:
        """Uses _predecessors cache (built from _P_model; O(|preds|) per call)."""
        for pred, _a in self._predecessors[s]:
            xi_pred = self.compute_bellman_residual(
                pred, self._P_model, self._r_model, self.gamma, self.V
            )
            self.xi[pred] = xi_pred
            if xi_pred > 1e-4:
                # Score = ξ + β_t·τ
                score = xi_pred + self.beta_t * self.tau[pred]
                v = self._version_xi.get(pred, -1) + 1
                self._version_xi[pred] = v
                heapq.heappush(self._Q_xi, (-score, v, pred))
                # Invalidate any H_q entry — state now has nonzero residual.
                if int(pred) in self._version_hq:
                    self._version_hq[int(pred)] += 1

    def notify_backup(
        self, s: int, adv_PS_row: np.ndarray, xi_s: float
    ) -> None:
        # s was just backed up; zero its residual so _rekey_residual_queue does
        # not re-insert it with stale priority on the next topology_update call.
        self.xi[s] = 0.0
        delta_pi = self.gta_policy_update(s, adv_PS_row)
        
        # Lazy topology update: accumulate delta_pi
        self._pending_delta_pi[s] += delta_pi
        self._has_pending_topo = True
        
        self._step_count += 1
        arnoldi_step = (self._step_count % self._T_arn == 0)
        topo_step = (self._step_count % self._T_topo == 0)

        if arnoldi_step:
            self._refresh_slem()
            if not topo_step:
                self._rekey_residual_queue()

        if topo_step:
            self.topology_update_lazy()

    def update_model(self, P: np.ndarray, r: np.ndarray) -> None:
        """Sync empirical model for predecessor lookup (DynaHost only).
        Does NOT rebuild LU or topology signals — those depend on oracle P."""
        self._P_model = P.copy()
        self._r_model = r.copy()
        self._build_predecessor_cache()
    def update_value(self, s: int, v_s: float) -> None:
        self.V[s] = v_s

    def reset(self) -> None:
        """Re-run full precompute on stored (self.P, self.r)."""
        self._P_model = self.P.copy()
        self._r_model = self.r.copy()
        self._init_topology()



    # ------------------------------------------------------------------
    # Convenience wrappers (backward compat + plan() shim)
    # ------------------------------------------------------------------

    def bellman_residual_state(self, s: int) -> float:
        return self.compute_bellman_residual(
            s, self._P_model, self._r_model, self.gamma, self.V
        )

    def td_residual_state(self, s: int) -> float:
        return self.compute_td_residual(
            s, self._P_model, self._r_model, self.gamma, self.V
        )

    def backup_state(self, s: int) -> float:
        return self.apply_bellman_backup(
            s, self._P_model, self._r_model, self.gamma, self.V
        )

    def reset_value(self) -> None:
        self.reset()

    def plan(self, K: int, r: np.ndarray | None = None) -> np.ndarray:
        """Run K Bellman backups (budget per host step). r triggers reward_update first.
        Backward-compat shim: GtaPS(P, r, gamma, **kwargs).plan(K, r=r_new) works.
        """
        if r is not None:
            self.reward_update(r - self.r)

        for _ in range(K):
            s = self.pop_and_prioritize()
            if s is None:
                break
            delta_V = self.apply_bellman_backup(
                s, self._P_model, self._r_model, self.gamma, self.V
            )
            # Q[:,s] over oracle P — O(|A|·|S|) per step, not O(|A|·|S|²)
            ev_s = (self.P[:, s, :] * self.V[None, :]).sum(axis=1)  # (|A|,)
            q_s = self._r_model[s] + self.gamma * ev_s               # (|A|,)
            adv_PS_row = q_s - q_s.max()                               # (|A|,)
            self.notify_backup(s, adv_PS_row, delta_V)
            self.push_predecessors(s, delta_V)

        self._last_beta = self.beta_t
        self._last_signal_ratio = float(
            np.sum(np.abs(self.beta_t * self.tau))
            / max(float(np.sum(self.xi)), 1e-6)
        )
        self._slem = self.slem
        return self.V.copy()

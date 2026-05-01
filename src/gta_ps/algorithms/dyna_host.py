"""
DynaHost — empirical-model (one-step TD) host, module-agnostic.

Uses P_hat built from experience. TD errors drive residuals — NOT
distributional Bellman. Syncs P_hat and V into the module after every
real_step so the module's predecessor lookup uses empirical data.
"""
from __future__ import annotations

import numpy as np

from .base import PrioritizedSweepingModule, TdMixin
from .empirical_model import CountModel, SampleModel, EmpiricalModel


class DynaHost(TdMixin):
    """
    Dyna (one-step TD) host.

    Parameters
    ----------
    n_states  : int
    n_actions : int
    gamma     : float
    eta_td    : float   TD learning rate
    module    : PrioritizedSweepingModule
    model_type : {'count', 'sample'}
        'count'  — CountModel (distributional, O(|A||S|²) memory).
        'sample' — SampleModel (most-recent transition, O(|A||S|) memory).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float,
        eta_td: float,
        module: PrioritizedSweepingModule,
        model_type: str = "count",
        r_init: np.ndarray | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.eta_td = eta_td
        self.module = module
        self.Q = np.zeros((n_actions, n_states))
        self.r_oracle = None if r_init is None else r_init.copy()
        if model_type == "sample":
            self.model = SampleModel(n_states, n_actions)
        elif model_type == "count":
            self.model = CountModel(n_states, n_actions)
        else:
            raise ValueError(f"model_type must be 'count' or 'sample', got {model_type!r}")

    def reward_update(self, r: np.ndarray) -> None:
        """Inject an oracle-known reward shift while keeping the empirical transition model."""
        module_r = getattr(self.module, "r", None)
        if module_r is None:
            raise RuntimeError("DynaHost.reward_update requires the module to expose its current reward vector")
        delta_r = r - np.asarray(module_r)
        self.r_oracle = r.copy()
        self.module.reward_update(delta_r)
        P_hat, _ = self.model.get_model()
        self.module.update_model(P_hat, self.r_oracle)


    def real_step(
        self, s: int, a: int, r: float, s_next: int
    ) -> None:
        """
        Process one real (s, a, r, s') interaction.
        TD residual — NOT distributional Bellman — drives priority.
        Syncs empirical model and V estimate into the module after every step.
        """
        self.model.update(s, a, s_next, r)
        P_hat, r_hat = self.model.get_model()
        reward_model = self.r_oracle if self.r_oracle is not None else r_hat

        # TD update (signed error)
        # We manually apply it here so we can clip if needed, though apply_td_backup is preferred.
        # Let's use apply_td_backup and just clip the Q values if they get too large.
        abs_delta = self.apply_td_backup(s, a, r, s_next, self.gamma, self.eta_td, self.Q)

        # Sync empirical model so push_predecessors uses P_hat
        self.module.update_model(P_hat, reward_model)
        # V(s) ≈ max_a Q(s,a)
        self.module.update_value(s, float(self.Q[:, s].max()))

        # Enqueue s for backup
        self.module.push_predecessors(s, abs_delta)


    def plan_steps(self, K: int) -> None:
        """
        Run K simulated Dyna planning steps using P_hat and TD updates.
        No distributional Bellman backup anywhere in this method.
        """
        # Cache model for the duration of planning to avoid redundant CSR conversions
        P_hat, r_hat = self.model.get_model()
        n_states = self.n_states
        gamma = self.gamma
        eta_td = self.eta_td
        
        # Pre-sample random values for epsilon-greedy if needed, but here a is argmax.
        # The main bottleneck is the per-backup module interaction and sampling.
        
        for _ in range(K):
            s = self.module.pop_and_prioritize()
            if s is None:
                break

            a = int(self.Q[:, s].argmax())

            # Sample next state from empirical model
            probs = P_hat[a, s]
            s_sim = int(np.random.choice(n_states, p=probs))
            reward_model = self.r_oracle if self.r_oracle is not None else r_hat
            r_sim = reward_model[s]

            # TD update
            abs_delta = self.apply_td_backup(s, a, r_sim, s_sim, gamma, eta_td, self.Q)

            # Sync updated value into module
            self.module.update_value(s, float(self.Q[:, s].max()))

            adv_PS_row = self.Q[:, s] - self.Q[:, s].max()
            self.module.notify_backup(s, adv_PS_row, abs_delta)
            self.module.push_predecessors(s, abs_delta)


    def reset(self) -> None:
        self.Q = np.zeros((self.n_actions, self.n_states))
        self.model.reset()
        self.module.reset()

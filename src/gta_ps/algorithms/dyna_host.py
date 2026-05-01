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
        # We don't sync the model here; plan_steps will do it.


    def real_step(
        self, s: int, a: int, r: float, s_next: int
    ) -> None:
        """
        Process one real (s, a, r, s') interaction.
        TD residual drives priority. Does NOT sync the entire model to the module
        every step (O(A*S^2) bottleneck); instead, only updates the local state value.
        """
        self.model.update(s, a, s_next, r)
        
        # TD update
        abs_delta = self.apply_td_backup(s, a, r, s_next, self.gamma, self.eta_td, self.Q)

        # Update local value in module
        self.module.update_value(s, float(self.Q[:, s].max()))

        # Enqueue s for backup. Note: push_predecessors might use a stale 
        # predecessor cache if real_step is called many times without plan_steps.
        # This is acceptable for performance; plan_steps syncs the cache.
        self.module.push_predecessors(s, abs_delta)


    def real_steps_batch(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray
    ) -> None:
        """
        Efficient batch update for high-volume interaction bursts.
        """
        self.model.update_batch(states, actions, next_states, rewards)
        for i in range(len(states)):
            s, a, r, sn = int(states[i]), int(actions[i]), float(rewards[i]), int(next_states[i])
            abs_delta = self.apply_td_backup(s, a, r, sn, self.gamma, self.eta_td, self.Q)
            self.module.update_value(s, float(self.Q[:, s].max()))
            self.module.push_predecessors(s, abs_delta)


    def plan_steps(self, K: int) -> None:
        """
        Run K simulated Dyna planning steps.
        Syncs the empirical model once before planning to ensure O(A*S^2)
        overhead is amortised across K backups.
        """
        # Sync model and rebuild predecessor cache once per planning burst
        P_hat, r_hat = self.model.get_model()
        reward_model = self.r_oracle if self.r_oracle is not None else r_hat
        self.module.update_model(P_hat, reward_model)
        
        n_states = self.n_states
        gamma = self.gamma
        eta_td = self.eta_td
        
        for _ in range(K):
            s = self.module.pop_and_prioritize()
            if s is None:
                break

            a = int(self.Q[:, s].argmax())

            # Sample next state from empirical model
            probs = P_hat[a, s]
            s_sim = int(np.random.choice(n_states, p=probs))
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

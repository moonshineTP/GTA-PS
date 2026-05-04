"""
DynaHost — empirical-model (one-step TD) host, module-agnostic.

Uses P_hat built from experience. TD errors drive residuals — NOT
distributional Bellman. Syncs P_hat and Q into the module after every
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
        rng: np.random.Generator | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.eta_td = eta_td
        self.module = module
        self.rng = rng if rng is not None else np.random.default_rng()
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
        Syncs only the affected transition row to the module (O(S) complexity).
        """
        self.model.update(s, a, s_next, r)
        p_row = self.model.get_row(s, a)
        r_model = self.r_oracle[s] if self.r_oracle is not None else float(r)
        
        self.module.update_model_transition(s, a, p_row, r_model)
        
        # TD update
        abs_delta = self.apply_td_backup(s, a, r, s_next, self.gamma, self.eta_td, self.Q)

        # Update local value in module
        self.module.update_value(s, float(self.Q[:, s].max()))
        self.module.push_predecessors(s, abs_delta)


    def real_steps_batch(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray
    ) -> None:
        """
        Efficient batch update for high-volume interaction bursts.
        """
        self.model.update_batch(states, actions, next_states, rewards)
        
        # Track which (s, a) pairs were updated to sync rows at the end
        updated_pairs = set()
        for i in range(len(states)):
            s, a, r, sn = int(states[i]), int(actions[i]), float(rewards[i]), int(next_states[i])
            updated_pairs.add((s, a))
            abs_delta = self.apply_td_backup(s, a, r, sn, self.gamma, self.eta_td, self.Q)
            self.module.update_value(s, float(self.Q[:, s].max()))
            self.module.push_predecessors(s, abs_delta)
            
        # Sync all updated rows once after the batch
        for s, a in updated_pairs:
            p_row = self.model.get_row(s, a)
            r_val = self.r_oracle[s] if self.r_oracle is not None else 0.0 # simplified for batch
            self.module.update_model_transition(s, a, p_row, r_val)


    def plan_steps(self, K: int) -> None:
        """
        Run K simulated Dyna planning steps.
        Now O(K*S) complexity: no O(A*S^2) synchronization call.
        """
        n_states = self.n_states
        gamma = self.gamma
        eta_td = self.eta_td
        
        # Use existing module model (kept in sync by real_step)
        P_hat = self.module._P_model
        r_model = self.module._r_model
        
        for _ in range(K):
            s = self.module.pop_and_prioritize()
            if s is None:
                break

            a = int(self.Q[:, s].argmax())

            # Sample next state from empirical model
            probs = P_hat[a, s]
            s_sim = int(self.rng.choice(n_states, p=probs))
            r_sim = r_model[s]

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

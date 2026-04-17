"""
Evaluation metrics for GTA-PS experiments.
"""
import numpy as np
from typing import Optional


def value_error_inf(V: np.ndarray, V_star: np.ndarray) -> float:
    """||V - V*||_inf"""
    return float(np.max(np.abs(V - V_star)))


def steps_to_epsilon(
    V_history: np.ndarray,
    V_star: np.ndarray,
    eps: float = 0.1,
) -> Optional[int]:
    """
    K_eps: first K such that ||V^K - V*||_inf <= eps * ||V*||_inf.
    V_history[k] = V after k backups.  Returns None if never reached.
    """
    threshold = eps * np.max(np.abs(V_star))
    for k, V in enumerate(V_history):
        if value_error_inf(V, V_star) <= threshold:
            return k
    return None


def cumulative_return(rewards: np.ndarray, gamma: float) -> float:
    """Sum_{t=0}^{T} gamma^t * r_t"""
    T = len(rewards)
    gammas = gamma ** np.arange(T)
    return float(np.dot(gammas, rewards))

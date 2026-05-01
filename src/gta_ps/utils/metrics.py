"""
Evaluation metrics for GTA-PS experiments.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class ValueMetric(ABC):
    """Abstract interface for value-function error metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for keys and filenames (e.g. 'l1')."""

    @property
    @abstractmethod
    def label(self) -> str:
        """LaTeX-formatted label for plots."""

    @abstractmethod
    def compute(self, V: np.ndarray, V_star: np.ndarray) -> float:
        """Calculate metric magnitude."""

    def __call__(self, V: np.ndarray, V_star: np.ndarray) -> float:
        return self.compute(V, V_star)


class LInfMetric(ValueMetric):
    @property
    def name(self): return "inf"
    @property
    def label(self): return r"$L_\infty$ (max)"
    def compute(self, V, V_star):
        return float(np.max(np.abs(V - V_star)))


class L2Metric(ValueMetric):
    @property
    def name(self): return "l2"
    @property
    def label(self): return r"$L_2$ (rms)"
    def compute(self, V, V_star):
        return float(np.sqrt(np.mean((V - V_star)**2)))


class L1Metric(ValueMetric):
    @property
    def name(self): return "l1"
    @property
    def label(self): return r"$L_1$ (mean)"
    def compute(self, V, V_star):
        return float(np.mean(np.abs(V - V_star)))


# Instances for convenient access
INF = LInfMetric()
L2  = L2Metric()
L1  = L1Metric()

REGISTRY: dict[str, ValueMetric] = {
    "inf": INF,
    "l2":  L2,
    "l1":  L1,
}

# --- Legacy Functional Wrappers ---

def value_error_inf(V: np.ndarray, V_star: np.ndarray) -> float:
    return INF(V, V_star)

def value_error_l2(V: np.ndarray, V_star: np.ndarray) -> float:
    return L2(V, V_star)

def value_error_l1(V: np.ndarray, V_star: np.ndarray) -> float:
    return L1(V, V_star)


# --- Utility Functions ---

def steps_to_epsilon(
    V_history: np.ndarray,
    V_star: np.ndarray,
    eps: float = 0.1,
    metric: ValueMetric = INF,
) -> Optional[int]:
    """
    K_eps: first K such that metric(V^K, V*) <= eps * metric(0, V*).
    """
    threshold = eps * metric(np.zeros_like(V_star), V_star)
    for k, V in enumerate(V_history):
        if metric(V, V_star) <= threshold:
            return k
    return None


def cumulative_return(rewards: np.ndarray, gamma: float) -> float:
    """Sum_{t=0}^{T} gamma^t * r_t"""
    T = len(rewards)
    gammas = gamma ** np.arange(T)
    return float(np.dot(gammas, rewards))


# --- Graph Smoothness Metrics (Lemma 6.9 / Poincaré) ---

def _stationary_dist(P: np.ndarray, n_iter: int = 2000, tol: float = 1e-10) -> np.ndarray:
    """Stationary distribution of row-stochastic P via power iteration."""
    n = P.shape[0]
    mu = np.ones(n, dtype=float) / n
    for _ in range(n_iter):
        mu_new = mu @ P
        if np.max(np.abs(mu_new - mu)) < tol:
            return mu_new
        mu = mu_new
    return mu


def graph_dirichlet_energy(f: np.ndarray, P_pi: np.ndarray) -> float:
    """
    Graph Dirichlet energy under the stationary-weighted reversibilized chain.

    E(f, f) = (1/2) sum_{s,s'} mu(s) P_tilde(s,s') (f(s) - f(s'))^2

    where P_tilde = (P_pi + P_pi^T) / 2  (additive reversibilization, Lemma 6.9).
    mu is the stationary distribution of P_pi.

    A small value means f is smooth on the transition graph.
    """
    P = np.asarray(P_pi, dtype=float)
    P_tilde = (P + P.T) / 2.0
    mu = _stationary_dist(P)
    diff_sq = (f[:, None] - f[None, :]) ** 2   # (n, n)
    return 0.5 * float(np.sum(mu[:, None] * P_tilde * diff_sq))


def graph_variance(f: np.ndarray, P_pi: np.ndarray) -> float:
    """
    Stationary-weighted variance of f.

    Var_mu(f) = sum_s mu(s) (f(s) - E_mu[f])^2

    This is the quantity bounded by Lemma 6.9:
      Var_mu(delta V^pi) <= 8 gamma^2 Delta_r^2 / ((1-gamma)^2 (1-SLEM)).
    """
    mu = _stationary_dist(np.asarray(P_pi, dtype=float))
    mean_f = float(np.dot(mu, f))
    return float(np.dot(mu, (f - mean_f) ** 2))

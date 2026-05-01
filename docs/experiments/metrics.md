# Evaluation Metrics

**File:** `src/gta_ps/utils/metrics.py`

## Value Error Metrics
- **`value_error_inf(V, V_star)`** ($L_\infty$) – primary error metric: `||V - V_star||_inf`.
- **`value_error_l2(V, V_star)`** ($L_2$) – root-mean-square error.
- **`value_error_l1(V, V_star)`** ($L_1$) – mean absolute error.

## Convergence Metrics
- **`steps_to_epsilon(V_history, V_star, eps=0.1)`** – first `K` where `||V^K - V*||_inf ≤ eps * ||V*||_inf`.

## Graph Smoothness & Geometry (Lemma 6.9)
- **`graph_dirichlet_energy(f, P_pi)`** – measures smoothness of a field `f` on the transition graph.
- **`graph_variance(f, P_pi)`** – stationary-weighted variance of field `f`.

## Statistical Aggregation
All experiments aggregate across seeds with a 95 % CI:
```python
mean = np.mean(vals)
ci95 = scipy.stats.sem(vals) * scipy.stats.t.ppf(0.975, df=len(vals) - 1)
```

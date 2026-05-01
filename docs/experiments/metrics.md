# Evaluation Metrics

**File:** `src/gta_ps/utils/metrics.py`

- **`value_error_inf(V, V_star)`** – primary error metric: `||V - V_star||_inf`.
- **`steps_to_epsilon(V_history, V_star, eps=0.1)`** – first `K` where `||V^K - V*||_inf ≤ eps * ||V*||_inf`.
- **`cumulative_return(rewards, gamma)`** – discounted return for online episodes.

All experiments aggregate across seeds with a 95 % CI based on the t‑distribution:
```python
mean = np.mean(vals)
ci95 = scipy.stats.sem(vals) * scipy.stats.t.ppf(0.975, df=len(vals) - 1)
```

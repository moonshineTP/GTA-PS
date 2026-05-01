# RQ1: Proactiveness Under Reward Shifts

**File:** `experiments/rq1_proactiveness.py`

## Research Question
Does GTA‑PS reduce post‑shift value error compared to Dyna‑PS when replanning budgets are tight (`K < t_mix(1/4)`)?

## Experimental Protocol (memo §8)
| Setting | Value |
|---------|-------|
| Environments | GARNET (`|S|=100,500`), FourRooms `k=1‑4` (≈104 states) |
| Budgets `K` | `|S|/4`, `|S|`, `4|S|` |
| Seeds | 1000 (full run); 20‑40 for quick checks |
| Shifts per seed | 5‑20 |
| Primary metric | `||V^K - V*||_inf` |
| Secondary metric | `K_ε` (see `metrics.py`) |

## Methods
| Test | Class | Key params |
|------|-------|------------|
| `dyna_ps` | `DynaPS` | residual‑only baseline |
| `gta_base` | `GtaPS` | α=0.5, β=1.0 |
| `gta_beta` | `GtaPS` | `adaptive_beta=True` |

All methods use a single agent per `(K, method)` and keep `V` warm‑started across reward shifts.

## Running
```bash
# quick smoke test
python experiments/rq1_proactiveness.py \
  --env garnet --n-states 100 --seeds 20 --n-shifts 5
```
Results are written to `results/rq1_<tag>.json`.

## Tuning‐Beta
See the dedicated tuning script: [[tune-beta.md]]

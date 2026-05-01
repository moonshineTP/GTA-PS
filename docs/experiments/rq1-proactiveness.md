# RQ1: Proactiveness & Convergence

**File:** `experiments/rq1_proactiveness.py`

## Research Question
Does GTA‑PS reduce value error faster than Standard‑PS (Dyna‑PS) in both stationary and non-stationary (reward shift) settings?

## Experimental Protocol (memo §8)
| Setting | Value |
|---------|-------|
| Environments | GARNET, FourRooms, Reacher |
| Budgets `K` | `|S|/4`, `|S|`, `4|S|` |
| Seeds | 20–100 |
| Shifts per seed | 5–20 (for timeline scripts) |
| Primary metric | `||V^K - V*||_inf` |

## Methods
| Test | Class | Key params |
|------|-------|------------|
| `dyna_ps` | `StandardPS` | residual‑only baseline |
| `gta_fixed` | `GtaPS` | `adaptive_beta=False`, `beta_mode='fixed'` |
| `gta_linear`| `GtaPS` | `adaptive_beta=True`, `beta_mode='linear'` |

## Running

### Stationary Convergence
```bash
python experiments/rq1_proactiveness.py --env garnet --seeds 20
```

### Timeline (Reward Shifts)
```bash
# Oracle DP setting
python experiments/rq1_dp_timeline.py --env fourrooms --seeds 10

# Learned Dyna setting
python experiments/rq1_dyna_timeline.py --env reacher --seeds 10
```

Results are written to `results/` as JSON files.

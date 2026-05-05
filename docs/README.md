# GTA-PS Code Documentation

This directory documents the `src/gta_ps` package and experiment scripts.

## Index

### Algorithms (`docs/algorithms/`)

| File | Contents |
|------|----------|
| [`gta-ps.md`](algorithms/gta-ps.md) | `GtaPS` class — variants, schedules, implementation notes |
| [`standard-ps.md`](algorithms/standard-ps.md) | `StandardPS` (DynaPS alias) + interface notes |
| [`spectral.md`](algorithms/spectral.md) | Laplacians, SLEM estimation, resolvent operators |

### Environments (`docs/environments/`)

| File | Contents |
|------|----------|
| [`overview.md`](environments/overview.md) | All envs — interface, constraints, setup (GARNET, FourRooms, Reacher, CarGoal) |

### Experiments (`docs/experiments/`)

| File | Contents |
|------|----------|
| [`rq1-proactiveness.md`](experiments/rq1-proactiveness.md) | RQ1 protocol, run commands, output format |
| [`metrics.md`](experiments/metrics.md) | `value_error_inf`, `steps_to_epsilon`, aggregation |

### Memo (`docs/memo/`)

The LaTeX memo (`GTA-PS-memo.tex`) in the repo root is the authoritative specification; the code is derived from it.

---

## Quick start

```bash
# Install (default stack: GARNET, FourRooms, MiniGrid, Reacher)
cd code/
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,reacher]"

# Smoke tests
pytest tests/test_smoke.py -q   # ~21 tests

# Quick RQ1 run
python experiments/rq1_convergence.py --n-states 100 --seeds 20

# Beta schedule comparison (RQ2)
python experiments/rq2_beta_magnitude.py
```

## Source layout

```
src/gta_ps/
├── algorithms/
│   ├── base.py              # Base classes (PrioritizedSweepingModule, Mixins)
│   ├── dp_host.py           # DP (oracle-model) host
│   ├── dyna_host.py         # Dyna (empirical-model) host
│   ├── empirical_model.py   # CountModel and SampleModel learners
│   ├── gta_ps.py            # GTA-PS algorithm (α/β variants)
│   ├── oracle_model.py      # Oracle transition/reward wrapper
│   └── standard_ps.py       # Standard PS (DynaPS alias)
├── envs/
│   ├── base.py              # TabularEnv interface
│   ├── garnet.py            # GARNET environment
│   ├── fourrooms.py         # FourRooms environment
│   ├── reacher.py           # Reacher-v5 wrapper
│   └── cargoal.py           # Safety-Gymnasium CarGoal wrapper
└── utils/
    ├── hyperparams.py       # Hyperparameter management
    ├── metrics.py           # Evaluation metrics
    └── spectral.py          # Laplacians and Arnoldi/SLEM
```

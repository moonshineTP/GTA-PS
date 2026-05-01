# GTA-PS Experiment Workspace

Code for all RQ experiments in the GTA-PS paper.

---

## Environment Setup

### Default stack (GARNET, FourRooms, MiniGrid, Reacher)

```bash
cd experiment/
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,reacher]"
```

Requires: Python ≥ 3.12, gymnasium ≥ 1.0, mujoco ≥ 3.1, minigrid ≥ 2.3.

### CarGoal (Safety-Gymnasium) — pinned stack

Safety-Gymnasium 1.0.0 is incompatible with gymnasium ≥ 1.0. Use a separate environment:

```bash
python -m venv .venv-cargoal && source .venv-cargoa/bin/activate
pip install "safety-gymnasium==1.0.0" "gymnasium==0.26.3" \
            "mujoco==2.3.3" "gymnasium-robotics==1.2.2" "pygame==2.1.0"
pip install -e ".[dev]"
```

Do **not** mix the two stacks.

---

## Smoke Tests

```bash
pytest tests/test_smoke.py -q    # 21 tests, ~4s
```

All 21 tests cover: GARNET, FourRooms, MiniGrid, Reacher, CountModelLearner, spectral utilities, DynaPS, GtaPS (base and adaptive).

---

## Reproducibility

### Seed discipline

- All experiments use `seeds: int` from the config file (e.g. `seeds: 100` → seeds 0..99).
- Environments accept a `seed` integer; algorithms are seeded by their environment.
- NumPy RNG: `np.random.default_rng(seed)` throughout — no global state.

### Config files

| Config | Environment | |S| | n_shifts | seeds |
|--------|-------------|-----|----------|-------|
| `configs/garnet.yaml` | GARNET | 100–2000 | 10 | 100 |
| `configs/minigrid_fourrooms.yaml` | MiniGrid-FourRooms-v0 | ~1040 | 10 | 50 |
| `configs/reacher.yaml` | Reacher-v5 | 256 | 10 | 100 |
| `configs/cargoa.yaml` | SafetyPointGoal1-v0 | 400 | 10 | 50 |

### Exact run commands

```bash
# RQ1: adaptation speed (proactiveness)
python experiments/rq1_proactiveness.py --config configs/garnet.yaml

# MiniGrid version of RQ1
python experiments/rq1_proactiveness.py --config configs/minigrid_fourrooms.yaml

# Reacher (requires reacher extra)
python experiments/rq1_proactiveness.py --config configs/reacher.yaml
```

Results are written to `results/{env}/{timestamp}/`.

## Windows Orchestration

From the repository root you can run the grouped Windows launchers:

```bat
run_sanity.cmd [flags]
run_rqs.cmd [flags]
run_ablations.cmd [flags]
```

Each group runs its member scripts sequentially. To run groups in parallel, launch the three `.cmd` files in separate terminals.

Only these flags are forwarded by the orchestrator, and only to scripts that support them:

```bat
--workers
--seeds
--K
--n-steps-per-shift
--n-shifts
```

Example:

```bat
run_rqs.cmd --workers 8 --seeds 10 --K 20 --n-steps-per-shift 40 --n-shifts 4
```

---

## Module Structure

```
src/gta_ps/
  algorithms/
    dyna_ps.py          # Dyna-PS: lazy-deletion priority queue
    gta_ps.py           # GTA-PS: convex mix (1-α)f_out + α*f_in; β sign rule
    model_learner.py    # CountModelLearner: Dirichlet count model
  envs/
    base.py             # TabularEnv ABC
    garnet.py           # Generative GARNET
    fourrooms.py        # Hand-built FourRooms tabular
    minigrid_tabular.py # MiniGrid-FourRooms-v0 exact extraction
    reacher.py          # Reacher-v5 discretized (np.digitize)
    cargoal.py           # SafetyPointGoal1-v0 discretized (pinned stack)
  utils/
    spectral.py         # out_laplacian, in_laplacian, slem_arnoldi, resolvent_forward
    metrics.py          # value_error_inf, steps_to_epsilon
```

### GTA-PS core vs. infrastructure

**Core** (algorithm logic — do not substitute library code):
- `algorithms/dyna_ps.py`, `algorithms/gta_ps.py`, `utils/spectral.py`

**Infrastructure** (wrappers over standard RL libraries):
- `envs/minigrid_tabular.py` — uses `minigrid` (pip-installable)
- `envs/reacher.py` — uses `gymnasium` + `mujoco`; discretization via `np.digitize`
- `envs/cargoal.py` — uses `safety-gymnasium`; pinned stack
- `algorithms/model_learner.py` — count-based; binning could use `sklearn.preprocessing.KBinsDiscretizer`

---

## Known Issues (as of 2026-04-17)

| # | Severity | Description | File |
|---|----------|-------------|------|
| 1 | Low | `model_learner.py` uses hand-written Dirichlet count model; `sklearn` alternative not explored | `algorithms/model_learner.py` |
| 2 | Low | `cargoal.py` `_obs_to_state` accesses `env.unwrapped.task.agent.pos` with fallback to `obs[0:2]`; fragile against safety-gymnasium API changes | `envs/cargoal.py` |
| 3 | Info | CarGoal cannot be instantiated in the default venv (gymnasium version conflict) | `envs/cargoal.py` |
| 4 | Info | `shift_reward()` in `ReacherDiscrete` resamples disturbance mode randomly; for controlled RQ4 sweeps use `disturbance` param directly | `envs/reacher.py` |

---

## Technical Debt

The following components are hand-written rather than using standard library equivalents. They are correct and tested, but noted for future integration work:

- **State discretization** (`reacher.py`, `cargoal.py`): `np.digitize` over `[-π, π]`. Consider `sklearn.preprocessing.KBinsDiscretizer` for joint/symmetry-aware binning.
- **Rollout collection** (`_collect_rollouts`): plain Python loop. Consider `stable-baselines3` `VecEnv` for parallel collection at larger scales.
- **Count model** (`model_learner.py`): Dirichlet-smoothed counts. Adequate for |S| ≤ 400; may need compression for larger spaces.

# Environments Overview

All environments expose a **common interface**:
- `P` (transition tensor `(|A|,|S|,|S|)`)
- `r` (reward vector `(|S|,)`)
- `shift_reward()`
- `value_iteration(r, tol, max_iter)`

## Host Modes

| Host | Environments | Model Availability |
|------|--------------|--------------------|
| **DP** (Exact) | [[garnet]] `src/gta_ps/envs/garnet.py`, [[fourrooms]] `src/gta_ps/envs/fourrooms.py` | known `P`
| **Dyna** (Learned) | [[reacher]] `src/gta_ps/envs/reacher.py`, [[cargoal]] `src/gta_ps/envs/cargoal.py` | empirical `P` via `CountModel`

## DP Environments
- **GARNET**: Generative tabular MDP (100–500 states, ≥10 actions). Reference Archibald et al. 1995.
- **FourRooms**: Deterministic 19×19 grid (≈384 states, 4 actions). Reward shifts in active rooms.

## Dyna Environments
- **Reacher**: 256‑state discretisation of 2‑joint Reacher (requires `gymnasium Reacher‑v5`, `mujoco`). Learns `P_hat` with `CountModelLearner`.
- **Cargoal**: 400‑state discretisation of SafetyPointGoal1‑v0 (requires `safety-gymnasium==1.0.0`, `gymnasium==0.26.3`). See the file for venv setup.

## Common Interface (example code)
```python
env.P          # (|A|, |S|, |S|)
env.r          # (|S|,)
env.n_states   # |S|
env.n_actions  # |A|
env.gamma

env.shift_reward()              # updates env.r
env.value_iteration(r, tol, max_iter)  # returns V*
```

See algorithm docs for host‑module interaction: [[standard-ps.md]], [[gta-ps.md]], [[spectral.md]]

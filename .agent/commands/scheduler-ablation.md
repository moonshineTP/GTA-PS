---
description: Design and implement controlled ablation comparing beta scheduler variants for GTA-PS^beta
allowed-tools: Read, Bash, Grep
---

# GTA-PS Scheduler Ablation

You are implementing and running a controlled ablation comparing β_t scheduler designs for GTA-PS^β.

## Background

The tuner (`code/misc/tuner.py`) reports GTA-PS^(α,β) converges **slower** than Dyna-PS. The current code implements a power schedule `β_t = β₁·SLEM^{β₂}` (always positive). The memo defines a linear shift schedule `β_t = β₁·((1−SLEM)−β₂)` which can be negative (dampening slow-mixing chains) and is the theoretically grounded formula for graceful degradation.

## Task
$ARGUMENTS

## Step 1 — Read the Current Implementation

```bash
grep -n "_beta_t\|beta_mode\|beta1\|beta2\|adaptive_beta" code/src/gta_ps/algorithms/gta_ps.py
```

## Step 2 — Add `beta_mode` Parameter to GtaPS

Add `beta_mode: str = "power"` to `__init__`. Update `_beta_t`:

```python
def _beta_t(self, slem: float) -> float:
    if not self.adaptive_beta:
        return self.beta_base
    if self.beta_mode == "fixed":
        return self.beta1
    if self.beta_mode == "linear":                          # memo-canonical
        return self.beta1 * ((1.0 - slem) - self.beta2)
    if self.beta_mode == "clip_linear":                     # diagnostic
        return max(0.0, self.beta1 * ((1.0 - slem) - self.beta2))
    return self.beta1 * (slem ** self.beta2)               # "power" (current default)
```

Guard negative heap priorities (clamp only for heap, not for value update):
```python
heap_priority = max(0.0, aggregate_s)
```

## Step 3 — Create `code/misc/scheduler_ablation.py`

Protocol:
- Environments: FourRooms k=1, GARNET |S|=100 supp=4
- Seeds: 20, reward shifts: 5
- Budgets: [|S|/4, |S|, 4|S|]
- Variants: `fixed`, `linear`, `clip_linear`, `power`, + Dyna-PS anchor
- Metrics: `‖V^K − V*‖∞` (mean ± 95% CI) + β_t trajectory per seed
- Output: `code/results/misc/scheduler_ablation/scheduler_ablation.json`

β_t logging: read `agent._last_beta` after each `plan()` call (requires M2 attribute from `/rq1-extend`).

## Step 4 — Analysis Criteria

1. Which scheduler achieves lowest `‖V^K − V*‖∞` at K = |S|/4?
2. Does `linear` outperform `clip_linear`? (Tests whether negative β_t dampening is beneficial.)
3. Is `linear` (memo-canonical) the best? If not, flag for memo revision.
4. Is graceful degradation satisfied? `gta_beta (linear)` ≥ `dyna_ps` at every K, or the heap clamp is insufficient.

## Step 5 — Verdict

Provide:
```
# Scheduler Ablation Results

## Summary table
| Scheduler | FourRooms K=S/4 | GARNET K=S/4 | Mean β_t sign |
|-----------|----------------|--------------|---------------|

## Verdict
[Which scheduler to adopt; linked to memo §3 claim]

## Required follow-up
[Changes needed in gta_ps.py, tuner.py, rq1_proactiveness.py]
```

## Critical Constraints

- β₂ in `linear` is a **threshold offset**, not a power exponent — do not conflate.
- Heap clamp is local — β_t itself is returned unclamped for trajectory analysis.
- SLEM is empirical — do not assert ordering across environments.

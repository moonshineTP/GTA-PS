---
description: Systematic correctness, performance, and memo-alignment audit of GTA-PS Python source
allowed-tools: Read, Bash, Grep, Glob
---

# GTA-PS Code Audit

You are auditing the GTA-PS Python implementation for correctness against the memo's mathematical definitions, numerical stability, computational complexity, and performance regressions.

## Task
$ARGUMENTS

If no specific target is given, audit all four checks below in order.

## Target Files

```
code/src/gta_ps/algorithms/gta_ps.py        # Core GTA-PS planner
code/src/gta_ps/utils/spectral.py           # Resolvent, Laplacian, SLEM
code/src/gta_ps/algorithms/dyna_ps.py       # Dyna-PS baseline
code/misc/time_benchmark.py                 # Timing harness
```

Always read line numbers first:
```bash
grep -n "def \|class " code/src/gta_ps/algorithms/gta_ps.py
grep -n "def \|class " code/src/gta_ps/utils/spectral.py
```

## CHECK 1 — Formula Alignment Against Memo §2–5

### 1a. β_t scheduler
The memo defines: `β_t = β₁ · ((1 − SLEM) − β₂)` — linear, can be negative.
- Flag if code uses `beta1 * slem**beta2` (power schedule) instead.
- Flag if no path produces β_t < 0 (graceful degradation requires it).

### 1b. Augmented priority
Memo Algorithm 1:
```
τ(s)        = (1 − α_t)·|Δf^out(s)| + α_t·|Δf^in(s)|   # topology signal only
aggregate(s) = ξ(s) + β_t · τ(s)                         # augmented priority
```
- Flag if `tau` variable conflates both ξ and the topology signal.
- Verify convex-mix coefficient and that ξ is Bellman residual (not augmented).

### 1c. Resolvent matrix
`f^out = ((1+ε)I − P)^{-1} r`; `f^in = (D^in − P^T + εI)^{-1} r`; `ε* = (1−γ)/γ`
- Verify `resolvent_forward` constructs `(1+eps)I - P`, not `(I-P) + eps*I`.
- Verify `in_laplacian` uses column sums for D^in.
- Check `auto` solver threshold aligns with benchmark timings.

### 1d. SLEM caching
- Flag if SLEM is recomputed every step when `slem_recompute_every` is unset.
- Confirm cached `_slem` invalidation matches recompute trigger.

## CHECK 2 — Numerical Stability

- Arnoldi 1e-8 tolerance: too tight for near-degenerate chains?
- `mods[1]` fallback in `slem_arnoldi`: returns SLEM≈1 incorrectly for identity-like chains?
- Resolvent conditioning when ε is small and P has eigenvalue near 1+ε.
- `fill_factor=1.0` in spilu: safe for dense GARNET matrices?
- Negative augmented priorities (β_t < 0): does heap logic remain correct?

## CHECK 3 — Computational Complexity

Cross-reference `code/misc/time_benchmark.py`:

| Operation | Expected | Flag if |
|-----------|---------|---------|
| Resolvent LU (n≤300) | O(n³) one-time | Called inside K-step loop |
| SLEM via Arnoldi | O(k·nnz) | Dense matvec used for sparse P |
| Per-backup Bellman | O(|A|) | Dense slice where sparse suffices |
| Policy derivation P_π | O(|A|·|S|) | Recomputed unnecessarily |

Flag any computation inside the backup loop that could be hoisted.

## CHECK 4 — Edge Cases

1. Cold start (`_f_out_prev is None`): `df_out = |f_out|` — verify magnitude scale.
2. Uniform reward: topology signal should dominate; confirm xi ≈ 0.
3. Negative β_t: `aggregate(s)` can go negative — heap entries must not invert ordering.

## Output Format

For each issue:
```
ISSUE [Check #, Severity: HIGH/MEDIUM/LOW]
File: <file>, Line ~<line>
Description: <what is wrong and why>
Memo reference: <§section or equation>
Fix: <concrete code change>
```

End with:
- Prioritized fix list (HIGH first)
- Estimated impact on tuner results for each HIGH issue
- Which fixes require re-running `time_benchmark.py`

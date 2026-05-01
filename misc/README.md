# GTA-PS Ablations & Structural Audits

Secondary ablation scripts and parameter audits that support the primary Research Questions (RQ1–4) and validate the theoretical claims in §6 of the memo. Each script isolates one structural property and produces a JSON result + PNG plot.

---

## Theory Validation Scripts (§6)

### `cumulative_tracking.py`
**Theory:** Theorem 6.10(ii) — cumulative tracking-error advantage  
**Claim:** TE^PS(N) − TE^GTA(N) ≥ N·Ā + o(N) → +∞ a.s. under K < t\_mix(1/4).  
**Setup:** On-policy Dyna + TD(0) on slow-mixing GARNET (n\_nbr\_rate=0.07, SLEM≈0.99, t\_mix≈143). K=40 << t\_mix. N=25 reward-shift episodes, T\_real=30 real steps per episode, T\_warmup=300 random steps.  
**Key outputs:**
- Panel 1: Per-episode L1 value error (log scale) — Standard-PS vs GTA-PS^β
- Panel 2: Cumulative TE(N) = Σ err\_t over episodes
- Panel 3: TE^PS(N) − TE^GTA(N) with linear fit slope Ā (should be positive and growing)

**Requires Dyna host** (DP has no episode structure — reward oracle cannot trigger the K < t\_mix regime).

```bash
python misc/cumulative_tracking.py --seeds 10 --n-episodes 25 --K 40
```

---

### `smoothness_ablation.py`
**Theory:** Lemma 6.9 (Poincaré / graph smoothness)  
**Claim:** The resolvent (L^out + η\*I)^{-1} acts as a graph low-pass filter: δf^out has lower graph-frequency content than δr; Var\_μ(δV^π) ≤ 8γ²Δ\_r² / ((1−γ)²·(1−SLEM)).

**Metric note:** Raw reward δr is highly sparse (only `reward_support` = 4 out of 100 states nonzero). Comparing absolute Dirichlet energy E(δr) vs E(δf) is therefore misleading — the resolvent diffuses a sparse signal to all states, necessarily increasing absolute amplitude. The **Rayleigh quotient** Rq(f) = E(f,f)/Var\_μ(f) is scale-invariant and directly measures graph frequency independent of sparsity or amplitude.

**Setup:** Oracle DpHost on GARNET (n=100). Time-series at n\_nbr\_rate=0.10 (SLEM≈0.98); cross-SLEM sweep over n\_nbr\_rate ∈ {0.05, 0.07, 0.10, 0.15, 0.25, 0.40}.

**Key outputs:**
- Panel 1: Rq(δr) vs Rq(δf) per shift — Rq(δf) << Rq(δr) confirms low-pass filtering (4× frequency reduction at SLEM≈0.98)
- Panel 2: Var\_μ(δf^out) per shift vs per-shift Lemma 6.9 bound using measured SLEM and ‖δr‖\_∞
- Panel 3: Var\_μ(δf^out) vs spectral gap (1−SLEM) across GARNET configs + 1/gap bound curve (Lemma 6.9 shape)

```bash
python misc/smoothness_ablation.py --seeds 10 --n-shifts 20 --n-shifts-cross 15 --workers auto
```

---

## Assumption Coverage Scripts

### `mixing_threshold_ablation.py`
**Theory:** Prop. `prop:mfpt`(i) — K-vs-t\_mix phase behaviour  
**Claim:** Whenever K < t\_mix(1/4), the MFPT-blocked set B\_K is non-empty and GTA-PS yields a positive cumulative tracking-error advantage; the advantage diminishes as K/t\_mix grows toward and past 1.  
**Setup:** DpHost (oracle transitions, no warm-up noise). Fixed K=40. Sweeps n\_nbr\_rate ∈ {0.05, 0.07, 0.10, 0.15, 0.25, 0.40} — resulting K/t\_mix(¼) ratios span 0.02–1.5, straddling the threshold. N=20 reward-shift episodes per seed.

**Why DpHost (not Dyna):** Fast-mixing GARNET is also easier to learn a model for, so Dyna confounds t\_mix with model quality. The oracle isolates the K effect cleanly.

**Key outputs:**
- Panel 1: Cumulative advantage TE^PS(N)−TE^GTA(N) vs episode N, one trajectory per mixing regime
- Panel 2: End-of-run advantage vs K/t\_mix ratio with vertical dashed line at K/t\_mix=1
- Panel 3: Per-episode L1 error for both variants across all n\_nbr\_rate values

**Note on seeds:** 5 seeds show high variance in the middle range (K/t\_mix ≈ 0.1–0.6); the clearest signal is at the extremes (K/t\_mix ≈ 0.02–0.04: advantage ≈ +0.3–0.6; K/t\_mix ≈ 1.5: advantage ≈ −1.0). Run ≥20 seeds for a monotone trend in the transition zone.

```bash
python misc/mixing_threshold_ablation.py --seeds 20 --n-episodes 30 --K 40
```

---

### `shift_rate_ablation.py`
**Theory:** Assumption 4b — stationary-ergodic reward shifts with mean amplitude Δ̄\_r > 0  
**Claim:** GTA-PS advantage is preserved across the full range of ergodic shift rates.  
**Setup:** Sweeps T (planning steps per shift window) ∈ {5, 10, 25, 50, 100}. Error measured at end of each T-step window. All three variants on GARNET n=100.

```bash
python misc/shift_rate_ablation.py --seeds 10 --n-shifts 10 --K 40
```

---

### `boltzmann_b_ablation.py`
**Theory:** Assumption 8 — Boltzmann temperature b ∈ (0, b\_max] ensures π^PS\_t(a|s) ≥ π\_act,min > 0  
**Claim:** GTA-PS advantage is robust across the full b range (not only for one specific temperature).  
**Setup:** Sweeps b ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} for gta\_base and gta\_beta. Standard-PS shown as horizontal baseline (b-independent).

```bash
python misc/boltzmann_b_ablation.py --seeds 10 --n-shifts 8 --K 40
```

---

## Algorithm & Hyperparameter Ablations

### `scheduler_ablation.py`
Compares three β\_t scheduler families (linear, power, log) against Standard-PS on FourRooms and GARNET. Sweeps β₂ across each scheduler's natural grid with β₁=1 fixed. Primary reference for choosing the linear scheduler as default.

### `beta_dynamics.py`
Traces SLEM, β\_t, and the β\_t·τ / ξ signal ratio across backup steps and reward shifts for each GTA-PS β-variant. Helps diagnose topology signal saturation or collapse.

### `gamma_ablation.py`
Sweeps discount factor γ ∈ [0.5, 0.98] to observe how the contraction coefficient affects convergence trajectory. L1 metric; dynamically rescales tuned hyperparameters.

### `mixing_ablation.py`
GTA-PS (linear scheduler) vs Standard-PS across slow (SLEM≈0.994), medium (SLEM≈0.933), and fast (SLEM≈0.327) mixing GARNET environments. Sweeps β₂ in the linear scheduler range and reports best power-scheduler config.

---

## Robustness / Edge-Case Scripts

### `periodicity_ablation.py`
Investigates robustness to periodic (bipartite) policy-induced chains where SLEM=1.0 (spectral gap collapses). Verifies GTA-PS degrades gracefully and does not diverge.

### `reward_support_ablation.py`
Sweeps reward density |supp(r\_t)| ∈ {1, 2, 4, 8, 16} on GARNET to quantify how sparsity of reward affects planning efficiency. Relevant to the sparse-reward regime in Assumption 4.

---

## Benchmarks

### `convergence_benchmark.py`
Measures backup steps to reach ‖V − V\*‖\_∞ < ε after each reward shift with warm initialization from the preceding shift. Agents: dyna\_ps, gta\_base, gta\_beta. Environments: garnet (n=100), fourrooms. Default ε=5.0.

### `time_benchmark.py`
Wall-clock timing per backup step. Median ± std over repeated runs (warmup excluded). Reports overhead of topology pre-computation (\_init\_topology) relative to Standard-PS.

---

## Utilities

### `tuner.py`
Hyperparameter search for GTA-PS variants. Outputs tuned configs consumed by `load_tuned_for_env("garnet")` / `load_tuned_for_env("fourrooms")`.

### `showcase.py`
**FourRooms graphical interface plug-in** — view interface for Dyna+Standard-PS vs Dyna+GTA-PS^β on FourRooms. Select mode via `--mode` flag.

**`--mode ablation`** (default): Multi-seed headless run. Saves a 6-panel PNG to `plots/fourrooms_ablation.png`:
- Panel 1: Per-episode mean L1 error with 95% CI
- Panel 2: Cumulative tracking error TE(N)
- Panels 3–6: Final value V(s) and priority/topology τ(s) heatmaps for each agent

**`--mode showcase`**: Single-seed animation saved as `plots/fourrooms_showcase.gif` (Agg+pillow, no display required). Each frame corresponds exactly to one **real host step** (one action + $K$ planning steps). 

**Key Features:**
- **Dynamic Window**: `--n-frames` captures the **last** $N$ steps of the simulation. Defaults to the entire navigation across all episodes if omitted.
- **Episode Tracking**: Displays the current Episode (Shift) number and the **cumulative reward** received within that episode.
- **Multi-Panel**: Side-by-side comparison of value functions $V(s)$ and priority/topology maps $\tau(s)$ for Standard-PS and GTA-PS$^{\beta}$.

```bash
# Ablation: parallel 30-seed run, 8 reward-shift episodes
python misc/showcase.py --mode ablation --seeds 30 --n-shifts 8 --K 40 --workers auto

# Animation: Capture the entire 10-episode run (100 steps/ep = 1000 frames)
python misc/showcase.py --mode showcase --n-shifts 10 --n-steps-per-shift 100 --K 40

# Animation: Capture ONLY the final 200 steps of a long run
python misc/showcase.py --mode showcase --n-shifts 20 --n-frames 200 --K 40
```

### `plot_academic.py`
**Shared plot style and palette** — canonical source of matplotlib rcParams and colour palette for all misc and experiment scripts. Moved here from `experiments/` since all ablations reference it.

Key exports:
- `apply_style()` — call once before `plt.figure()` to apply canonical rcParams
- `PALETTE` — `{variant: hex_color}` for standard_ps / gta_base / gta_beta
- `LABELS` — `{variant: latex_string}` display names
- `plot_rq2()`, `plot_rq3()` — academic-quality experiment result renderers

All misc scripts import `from plot_academic import apply_style, PALETTE`.

---

## Technical Standards

All statistical scripts follow:
1. **Parallel execution** via `RunManager(n_workers="auto")` (tqdm progress bars included)
2. **95% CI** via Student-t over seeds (`scipy.stats.sem × t.ppf(0.975, df=seeds−1)`)
3. **Dual output**: standardized JSON in `results/misc/<name>/` + PNG in `plots/`
4. **Thread pinning**: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
5. **Shared style**: `from plot_academic import apply_style, PALETTE` — call `apply_style()` before first `plt.figure()`
6. **Config print**: each script prints a summary table of hyperparameters before the sweep loop

```bash
# Example: run the Thm 6.10 validation with 10 seeds
python misc/cumulative_tracking.py --seeds 10 --workers auto

# Example: run Lemma 6.9 smoothness ablation
python misc/smoothness_ablation.py --seeds 10 --n-shifts 20 --n-shifts-cross 15 --workers auto
```

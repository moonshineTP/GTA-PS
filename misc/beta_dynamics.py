"""
misc/beta_dynamics.py

Tracks SLEM, β_t, and β_t·τ / ξ signal ratio across backup steps and
reward shifts for each GTA-PS β-variant on FourRooms and GARNET.

Output: results/misc/beta_dynamics/dynamics.json  (raw trace)
        printed table per environment
"""
import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    # Try reaching up to find the real project root (e.g. if compiled in artifacts/dist/)
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
sys.path.insert(0, str(ROOT / "src"))

from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.algorithms import make_agent
from gta_ps.utils.hyperparams import load_tuned_for_env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VARIANTS  = ["gta_beta"]
N_SHIFTS  = 8          # reward shifts to simulate per seed
K_PER_STEP = 20        # Bellman backups per planning call
N_PLAN_STEPS = 5       # plan() calls between shifts (to see intra-shift dynamics)
GAMMA     = 0.95


def _make_env(env_name: str, seed: int):
    if env_name == "fourrooms":
        return FourRoomsTabular(k=1, gamma=GAMMA, seed=seed)
    return GarnetEnv(n_states=100, n_actions=5, branching=3,
                     reward_support=4, n_nbr_rate=0.15, gamma=GAMMA, seed=seed)


def _run_trace(env_name: str, variant: str) -> List[Dict]:
    """Return list of per-step records: shift_idx, plan_step, slem, beta_t, signal_ratio."""
    tuned = load_tuned_for_env(env_name).get(variant) or {}
    env   = _make_env(env_name, seed=0)
    agent = make_agent(variant, P=env.P, r=env.r, gamma=GAMMA, q=20, **tuned)
    mod   = agent.module

    records = []

    for shift_idx in range(N_SHIFTS):
        r_old = env.r.copy()
        r_new = env.shift_reward()
        agent.plan(K=K_PER_STEP, r=r_new)   # first plan call absorbs the shift

        for step in range(N_PLAN_STEPS):
            agent.plan(K=K_PER_STEP)
            records.append({
                "shift":        shift_idx,
                "plan_step":    step,
                "slem":         round(float(mod._slem), 6),
                "beta_t":       round(float(mod._last_beta), 6),
                "signal_ratio": round(float(mod._last_signal_ratio), 6),
                "one_minus_slem": round(1.0 - float(mod._slem), 6),
                "beta2":        round(float(mod.beta2), 6),
                "beta1":        round(float(mod.beta1), 6),
            })

    return records


def _summarise(records: List[Dict], env_name: str, variant: str) -> Dict:
    slems       = [r["slem"]         for r in records]
    betas       = [r["beta_t"]       for r in records]
    ratios      = [r["signal_ratio"] for r in records]
    one_minus   = [r["one_minus_slem"] for r in records]
    beta2s      = [r["beta2"]        for r in records]

    zero_beta_pct = 100.0 * sum(b == 0.0 for b in betas) / len(betas)
    feasible     = [om - b2 for om, b2 in zip(one_minus, beta2s)]

    return {
        "env": env_name, "variant": variant,
        "slem_mean":         round(float(np.mean(slems)),    5),
        "one_minus_slem":    round(float(np.mean(one_minus)), 6),
        "beta2":             round(float(np.mean(beta2s)),    6),
        "feasibility_gap":   round(float(np.mean(feasible)),  6),  # (1-SLEM)-beta2; <0 → dead zone
        "beta_t_mean":       round(float(np.mean(betas)),    5),
        "beta_t_max":        round(float(np.max(betas)),     5),
        "pct_beta_zero":     round(zero_beta_pct, 1),
        "signal_ratio_mean": round(float(np.mean(ratios)),   5),
    }


def main():
    all_traces: Dict = {}
    summaries: List[Dict] = []

    envs = ["fourrooms", "garnet"]

    for env_name in envs:
        all_traces[env_name] = {}
        print(f"\n{'='*70}")
        print(f"  Environment: {env_name}")
        print(f"{'='*70}")
        print(f"  {'variant':<12}  {'SLEM':>8}  {'1-SLEM':>8}  {'beta2':>8}  "
              f"{'gap':>8}  {'β_t mean':>9}  {'β_t max':>8}  {'%β=0':>6}  {'sig/ξ':>7}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  "
              f"{'-'*8}  {'-'*9}  {'-'*8}  {'-'*6}  {'-'*7}")

        for variant in VARIANTS:
            try:
                records = _run_trace(env_name, variant)
            except Exception as e:
                print(f"  {variant:<12}  ERROR: {e}")
                continue

            all_traces[env_name][variant] = records
            s = _summarise(records, env_name, variant)
            summaries.append(s)
            print(
                f"  {variant:<12}  "
                f"{s['slem_mean']:>8.5f}  "
                f"{s['one_minus_slem']:>8.6f}  "
                f"{s['beta2']:>8.6f}  "
                f"{s['feasibility_gap']:>+8.6f}  "
                f"{s['beta_t_mean']:>9.5f}  "
                f"{s['beta_t_max']:>8.5f}  "
                f"{s['pct_beta_zero']:>6.1f}%  "
                f"{s['signal_ratio_mean']:>7.4f}"
            )

    out = (ROOT / "results" / "misc" / "beta_dynamics" / "dynamics.json").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"traces": all_traces, "summaries": summaries}, f, indent=2)
    print(f"\nSaved to {out}")

    # --- Insight report ---
    print("\n" + "="*70)
    print("  INSIGHT REPORT")
    print("="*70)
    for s in summaries:
        gap  = s["feasibility_gap"]
        dead = s["pct_beta_zero"]
        env  = s["env"]
        var  = s["variant"]
        bt   = s["beta_t_mean"]

        if gap < 0:
            status = f"DEAD ZONE  gap={gap:+.6f}  → β_t forced to 0 on {dead:.0f}% of steps"
        elif gap < 1e-4:
            status = f"MARGINAL   gap={gap:+.6f}  → β_t barely positive ({bt:.5f} mean)"
        else:
            status = f"OK         gap={gap:+.6f}  → β_t active (mean={bt:.5f})"
        print(f"  [{env:<10s}] {var:<12s}: {status}")

    print("""
Key insight
-----------
β_t = max(0, β₁·((1−SLEM) − β₂))

The dead-zone condition is  β₂ ≥ (1−SLEM).

For a slow-mixing chain (SLEM → 1):
  • (1−SLEM) → 0, so even a tiny β₂ kills the signal.
  • The topology signal is WEAKEST precisely when SLEM is highest
    — i.e. when the chain is hardest to mix and needs the most help.
  • This is the correct qualitative direction: a slow-mixing chain has
    unreliable spectral estimates, so GTA-PS should be conservative.
    BUT β_t = 0 entirely abandons the topology signal rather than
    gracefully attenuating it.

Recommended fix (beyond the β₂-fraction fix already applied):
  Option A — soft floor: β_t = max(ε_floor, β₁·((1−SLEM) − β₂))
             where ε_floor > 0 (e.g. 1e-3) ensures H_q is always
             populated and the topology frontier is never abandoned.
  Option B — relative schedule: tune β₂ as a fraction of (1−SLEM)
             (already implemented via beta2_frac in the tuner).
             With beta2_frac ∈ [0, 1), β_t > 0 always.
  Option C — beta1 compensation: scale β₁ ∝ 1/(1−SLEM) so the
             topology weight stays in a fixed absolute range even
             as SLEM → 1.  Requires bounding β₁ to prevent explosion.
""")


if __name__ == "__main__":
    main()

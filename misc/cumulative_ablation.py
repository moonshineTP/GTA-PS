"""
misc/cumulative_tracking.py

On-policy Dyna ablation: cumulative L1 tracking error across reward-shift episodes.

Validates Theorem 6.10(ii):
  TE^PS(N) - TE^GTA(N)  >=  N * A_bar + o(N)  ->  +inf  a.s.

Protocol per seed:
  Warm-up : T_WARMUP real steps with random policy to populate P_hat ~= P.
  Episodes: for t in 1..N_EPISODES:
    1. env.shift_reward() -> r_new
    2. Signal update: host.reward_update(r_new) so topology refreshes and the host cache stays aligned.
    3. Run T_REAL real steps (epsilon-greedy) + K planning steps per step.
    4. V_est = max_a Q[:, s]  ;  V* = env.value_iteration(r_new)
    5. err_t = L1(V_est, V*)
  Cumulate: TE(n) = sum_{t=1}^{n} err_t

Environment: slow-mixing GARNET (n_nbr_rate=0.07, SLEM~0.99)
so that K << t_mix(1/4) and Theorem 6.10's regime is active.

IMPORTANT: Epsilon-greedy (EPS_GREEDY=0.15) is vital for cumulative reward.
Without it, GTA-PS over-exploits high-priority states and misses reward
opportunities elsewhere. This applies to all scripts optimizing cumulative
reward (see docs/algorithms/gta-ps.md for details).

Output:
  plots/cumulative_tracking.png
  results/misc/cumulative_tracking/tracking.json
"""
from __future__ import annotations

import sys
import os
import argparse
import json
import numpy as np
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    # Try reaching up to find the real project root (e.g. if compiled in artifacts/dist/)
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))

from gta_ps.envs.garnet import GarnetEnv
from gta_ps.algorithms import make_agent
from gta_ps.utils.metrics import value_error_l1
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager
from plot_academic import apply_style, get_palette, LABELS as AGENT_LABELS

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    plt = None

# Slow-mixing GARNET: n_nbr_rate ≈ 0.07 → SLEM ~ 0.99 → t_mix >> K
N_STATES    = 75            # default, overridden by --n-states
N_ACTIONS   = 4
BRANCHING   = 3
N_NBR_RATE  = 0.07   # slow mixing;
REWARD_SUP  = 4
GAMMA       = 0.95

N_EPISODES  = 20
T_WARMUP    = 200           # real steps before episodes begin
T_REAL      = 50            # real steps per episode (after shift)
K           = 20            # planning steps per real step
EPS_GREEDY  = 0.2    # epsilon-greedy exploration rate

VARIANTS    = ["standard_ps", "gta_beta"]

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    n_episodes: int,
    t_warmup: int,
    t_real: int,
    K: int,
    gamma: float,
    method_cfgs: dict,
    n_states: int = N_STATES,
    **_ignored,
) -> dict:
    env      = GarnetEnv(n_states=n_states, n_actions=N_ACTIONS, branching=BRANCHING,
                         n_nbr_rate=N_NBR_RATE, reward_support=REWARD_SUP,
                         gamma=gamma, seed=seed)
    sim_rng  = np.random.default_rng(seed + 0xFACE_CAFE)
    traj_rngs = {
        v: np.random.default_rng(np.random.SeedSequence([seed, idx, 0xFACE]))
        for idx, v in enumerate(VARIANTS)
    }

    agents = {}
    for idx, v in enumerate(VARIANTS):
        kw = method_cfgs.get(v, {}).copy() if v != "standard_ps" else {}
        agents[v] = make_agent(
            v, host="dyna", P=env.P, r=env.r, gamma=gamma, q=20,
            rng=np.random.default_rng(np.random.SeedSequence([seed, idx, 0xC0DA])),
            **kw
        )

    cur_s = {v: int(sim_rng.integers(n_states)) for v in VARIANTS}

    def _real_step_plan(v: str, s: int, r_vec: np.ndarray) -> int:
        """One real step + K plan steps. Returns next state."""
        traj_rng = traj_rngs[v]
        if traj_rng.random() < EPS_GREEDY:
            a = int(traj_rng.integers(N_ACTIONS))
        else:
            a = int(agents[v].Q[:, s].argmax())
        probs  = env.P[a, s]
        s_next = int(traj_rng.choice(n_states, p=probs))
        r_env  = float(r_vec[s])        # state-based reward at current state
        agents[v].real_step(s, a, r_env, s_next)
        agents[v].plan_steps(K)
        return s_next

    # ── Warm-up (random exploration) ────────────────────────────────────────
    r0 = env.r.copy()
    for _ in range(t_warmup):
        for v in VARIANTS:
            traj_rng = traj_rngs[v]
            s      = cur_s[v]
            a      = int(traj_rng.integers(N_ACTIONS))
            probs  = env.P[a, s]
            s_next = int(traj_rng.choice(n_states, p=probs))
            agents[v].real_step(s, a, float(r0[s]), s_next)
            agents[v].plan_steps(K)
            cur_s[v] = s_next

    # ── Episodes ─────────────────────────────────────────────────────────────
    per_ep:   dict[str, list[float]] = {v: [] for v in VARIANTS}
    cumul:    dict[str, list[float]] = {v: [0.0] for v in VARIANTS}

    for _ in range(n_episodes):
        r_new  = env.shift_reward()
        V_star = env.value_iteration(r_new)

        # Notify the host so its reward cache stays aligned with the module.
        for agent in agents.values():
            agent.reward_update(r_new)

        # T_REAL real steps in new environment
        for _ in range(t_real):
            for v in VARIANTS:
                cur_s[v] = _real_step_plan(v, cur_s[v], r_new)

        # End-of-episode error
        for v in VARIANTS:
            V_est = agents[v].Q.max(axis=0)
            err   = value_error_l1(V_est, V_star)
            per_ep[v].append(err)
            cumul[v].append(cumul[v][-1] + err)

    return {"per_ep": per_ep, "cumul": cumul}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    per_ep_mean, per_ep_ci, cumul_mean, cumul_ci,
    n_episodes: int, seeds: int, K: int, t_real: int, n_states: int, out_path: str,
) -> None:
    if plt is None:
        print("matplotlib not available - skipping plot.")
        return

    apply_style()
    p_line = get_palette("line")

    episodes = np.arange(1, n_episodes + 1)
    cumul_ep = np.arange(0, n_episodes + 1)

    # Difference: TE^PS - TE^GTA
    diff_mean = np.array(cumul_mean["standard_ps"]) - np.array(cumul_mean["gta_beta"])
    diff_ci   = np.sqrt(
        np.array(cumul_ci["standard_ps"]) ** 2 +
        np.array(cumul_ci["gta_beta"])    ** 2
    )

    fig = plt.figure(figsize=(15, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel 1: Per-episode error ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    for v in VARIANTS:
        m = np.array(per_ep_mean[v])
        c = np.array(per_ep_ci[v])
        ax1.plot(episodes, m, label=AGENT_LABELS.get(v, v), color=p_line.get(v, "#444444"), linewidth=1.8)
        ax1.fill_between(episodes, m - c, m + c, color=p_line.get(v, "#444444"), alpha=0.15)
    ax1.set_xlabel("Episode (reward shift index)")
    ax1.set_ylabel(r"L$_1$ error at episode end")
    ax1.set_title("Per-episode value error")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.25)
    ax1.legend(fontsize=9)

    # ── Panel 2: Cumulative TE(N) ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for v in VARIANTS:
        m = np.array(cumul_mean[v])
        c = np.array(cumul_ci[v])
        ax2.plot(cumul_ep, m, label=AGENT_LABELS.get(v, v), color=p_line.get(v, "#444444"), linewidth=1.8)
        ax2.fill_between(cumul_ep, m - c, m + c, color=p_line.get(v, "#444444"), alpha=0.15)
    ax2.set_xlabel("Episode N")
    ax2.set_ylabel(r"$\mathrm{TE}(N) = \sum_{t=1}^{N} \mathrm{err}_t$")
    ax2.set_title("Cumulative tracking error TE(N)")
    ax2.grid(True, ls="--", alpha=0.25)
    ax2.legend(fontsize=9)

    # ── Panel 3: TE^PS(N) - TE^GTA(N)  [Thm 6.10(ii)] ──────────────────────
    ax3 = fig.add_subplot(gs[2])
    c_adv = p_line["gta_base"]  # Use GTA-Base sky blue for advantage curve
    ax3.plot(cumul_ep, diff_mean, color=c_adv, linewidth=2.2,
             label="GTA Advantage")
    ax3.fill_between(cumul_ep, diff_mean - diff_ci, diff_mean + diff_ci,
                     color=c_adv, alpha=0.18)

    # Reference: zero line
    ax3.axhline(0, color="gray", ls=":", linewidth=1)
    # Linear fit for slope annotation
    if n_episodes >= 5:
        x_fit = cumul_ep[1:]
        coeff = np.polyfit(x_fit, diff_mean[1:], 1)
        ax3.plot(cumul_ep, np.polyval(coeff, cumul_ep), ls="--",
                 color="#ff7f0e", linewidth=1.4,
                 label="Linear Fit")
    ax3.set_xlabel("Episode N")
    ax3.set_ylabel(r"$\mathrm{TE}^{\mathrm{PS}}(N) - \mathrm{TE}^{\mathrm{GTA}}(N)$")
    ax3.set_title("Cumulative advantage (Thm 6.10 ii)")
    ax3.grid(True, ls="--", alpha=0.25)
    ax3.legend(fontsize=8)

    fig.suptitle(
        rf"Cumulative Tracking Ablation — Dyna/TD(0), GARNET $|S|={n_states}$, "
        rf"SLEM$\approx$0.99, $K={K}$, $T_{{real}}={t_real}$, $N_{{seeds}}={seeds}$",
        fontsize=11,
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cumulative Dyna tracking ablation - validates Thm 6.10(ii)"
    )
    parser.add_argument("--n-episodes", type=int,   default=N_EPISODES)
    parser.add_argument("--t-warmup",   type=int,   default=T_WARMUP)
    parser.add_argument("--t-real",     type=int,   default=T_REAL)
    parser.add_argument("--K",          type=int,   default=K)
    parser.add_argument("--gamma",      type=float, default=GAMMA)
    parser.add_argument("--n-states",    type=int,   default=N_STATES)
    parser.add_argument("--seeds",      type=int,   default=5)
    parser.add_argument("--workers",    default="auto")
    parser.add_argument("--out-dir",
                        default=str(ROOT / "results/misc/cumulative_tracking"))
    parser.add_argument("--plot-dir",   default=str(ROOT / "plots"))
    args = parser.parse_args()

    from scipy import stats as sp_stats

    tuned  = load_tuned_for_env("garnet")
    runner = RunManager(n_workers=args.workers)

    print(f"Running {args.seeds} seeds x {args.n_episodes} episodes "
          f"(warm={args.t_warmup}, T_real={args.t_real}, K={args.K}) ...")

    raw = runner.run(
        _run_seed, range(args.seeds),
        dict(n_episodes=args.n_episodes, t_warmup=args.t_warmup,
             t_real=args.t_real, K=args.K, gamma=args.gamma,
             method_cfgs=tuned, n_states=args.n_states),
        desc="cumulative_tracking",
    )

    # ── Aggregate ──────────────────────────────────────────────────────────
    def _agg(key: str, v: str):
        mat = np.array([r[key][v] for r in raw])
        mean = mat.mean(axis=0)
        ci95 = sp_stats.sem(mat, axis=0) * sp_stats.t.ppf(0.975, df=len(mat) - 1)
        return mean.tolist(), ci95.tolist()

    per_ep_mean: dict = {}
    per_ep_ci:   dict = {}
    cumul_mean:  dict = {}
    cumul_ci:    dict = {}

    print(f"\n{'variant':<14}  {'mean per-ep err':>16}  "
          f"{'cumul TE(N)':>12}  {'TE_PS-TE_GTA':>13}")
    print("-" * 60)

    ps_cumul_end = None
    for v in VARIANTS:
        per_ep_mean[v], per_ep_ci[v] = _agg("per_ep", v)
        cumul_mean[v],  cumul_ci[v]  = _agg("cumul",  v)
        print(f"  {v:<14}  {float(np.mean(per_ep_mean[v])):>16.4f}  "
              f"{cumul_mean[v][-1]:>12.4f}")
        if v == "standard_ps":
            ps_cumul_end = cumul_mean[v][-1]

    diff_end = ps_cumul_end - cumul_mean["gta_beta"][-1]
    print(f"\n  TE^PS(N) - TE^GTA(N) at N={args.n_episodes}: {diff_end:.4f} "
          f"({'GROWING' if diff_end > 0 else 'NOT GROWING - CHECK'})")

    # ── Persist ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "n_states":   args.n_states,
            "n_nbr_rate": N_NBR_RATE,
            "gamma":      args.gamma,
            "K":          args.K,
            "t_real":     args.t_real,
            "t_warmup":   args.t_warmup,
            "n_episodes": args.n_episodes,
            "seeds":      args.seeds,
        },
        "per_ep_mean":  per_ep_mean,
        "per_ep_ci":    per_ep_ci,
        "cumul_mean":   cumul_mean,
        "cumul_ci":     cumul_ci,
    }
    out_path = out_dir / "cumulative_ablation.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_path}")

    plot_path = str(Path(args.plot_dir).resolve() / "cumulative_ablation.png")
    plot_results(
        per_ep_mean, per_ep_ci, cumul_mean, cumul_ci,
        args.n_episodes, args.seeds, args.K, args.t_real, args.n_states, plot_path,
    )


if __name__ == "__main__":
    main()

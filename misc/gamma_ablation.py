"""
RQ1 Ablation: Effect of Discount Factor (gamma) on Convergence.

This script sweeps gamma in [0.5, 0.98] to observe how the contraction 
coefficient affects the convergence trajectory of GTA-PS vs Standard PS.
Uses L1 metric and dynamically scales tuned hyperparameters.
"""
import os
import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Pin BLAS threads
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
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms import make_agent
from gta_ps.utils.metrics import value_error_l1
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager
from plot_academic import apply_style, get_palette, LABELS as AGENT_LABELS

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_gamma_seed(
    gamma: float,
    env_name: str,
    n_states: int | None,
    k_fourrooms: int | None,
    seed: int,
    n_steps: int,
    K: int,
    method_cfgs: dict,
    **ignored,
) -> dict:
    """Run convergence for a specific gamma and seed."""
    if env_name == "garnet":
        env = GarnetEnv(n_states=n_states, seed=seed, gamma=gamma)
    else:
        env = FourRoomsTabular(k=k_fourrooms, gamma=gamma, seed=seed)
    
    variants = ["standard_ps", "gta_base", "gta_beta"]
    r = env.r
    V_star = env.value_iteration(r)
    
    agents = {}
    for v in variants:
        kw = method_cfgs.get(v, {}).copy()
        
        # Scaling correction for eta (resolvent epsilon)
        # Tuned parameters are for gamma=0.95. If eta exists, scale it.
        if "eta" in kw and kw["eta"] is not None:
            tuned_gamma = 0.95
            eps_tuned = (1.0 - tuned_gamma) / tuned_gamma
            eps_current = (1.0 - gamma) / gamma
            # Scale eta proportionally to the shift in eps*
            kw["eta"] = kw["eta"] * (eps_current / eps_tuned)

        agents[v] = make_agent(v, host="dp", P=env.P, r=r, gamma=gamma, q=20, **kw)
    
    traces = {v: [] for v in variants}
    for _ in range(n_steps):
        for v in variants:
            V = agents[v].plan(K=K)
            # Switch to L1 per user instruction
            traces[v].append(float(value_error_l1(V, V_star)))
            
    return traces

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation(agg_results, gammas, K, seeds, n_steps, env_label, out_path):
    if plt is None: return
    apply_style()
    
    variants = ["standard_ps", "gta_base", "gta_beta"]
    
    fig, axes = plt.subplots(1, len(variants), figsize=(18, 5), sharey=True)
    steps_arr = np.arange(n_steps)
    
    # Color map for gammas
    cmap = plt.get_cmap("viridis")
    gamma_colors = {g: cmap(i / (len(gammas)-1)) for i, g in enumerate(gammas)}

    for i, v in enumerate(variants):
        ax = axes[i]
        for g in gammas:
            res = agg_results[str(g)][v]
            mean = np.array(res["mean"])
            ci = np.array(res["ci95"])
            ax.plot(steps_arr, mean, label=rf"$\gamma$={g}", color=gamma_colors[g], linewidth=1.5)
            ax.fill_between(steps_arr, mean - ci, mean + ci, color=gamma_colors[g], alpha=0.1)
        
        ax.set_yscale("log")
        ax.set_title(f"{AGENT_LABELS.get(v, v)}")
        ax.set_xlabel("Host Step")
        if i == 0: ax.set_ylabel(r"Value Error ($L_1$)")
        ax.grid(True, which="both", ls="-", alpha=0.1)
        if i == len(variants) - 1:
            ax.legend(title="Discount Factor", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(f"Gamma Ablation on {env_label} (N={seeds}, K={K})\nMetric: L1 Error with Scaled Hyperparameters", fontsize=14, y=1.05)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation: Effect of Gamma on GTA-PS Convergence")
    parser.add_argument("--env", default="garnet", choices=["garnet", "fourrooms"])
    parser.add_argument("--n-states", type=int, default=100)
    parser.add_argument("--k-fourrooms", type=int, default=1)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--plot-dir", default=str(ROOT / "plots"))

    args = parser.parse_args()
    
    gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    
    method_cfgs = load_tuned_for_env(args.env)
    
    runner = RunManager(n_workers=args.workers)
    
    final_results = {}
    env_label = f"Garnet(S={args.n_states})" if args.env == "garnet" else f"FourRooms(k={args.k_fourrooms})"
    
    for g in gammas:
        print(f"Running ablation for gamma={g}...")
        fn_kwargs = dict(
            gamma=g, env_name=args.env, n_states=args.n_states, 
            k_fourrooms=args.k_fourrooms, n_steps=args.n_steps, K=args.K,
            method_cfgs=method_cfgs
        )
        seed_results = runner.run(run_gamma_seed, range(args.seeds), fn_kwargs)
        
        # Aggregate
        from scipy import stats
        variants = list(seed_results[0].keys())
        summary = {}
        for v in variants:
            mat = np.array([r[v] for r in seed_results])
            summary[v] = {
                "mean": mat.mean(axis=0).tolist(),
                "ci95": (stats.sem(mat, axis=0) * stats.t.ppf(0.975, df=len(mat)-1)).tolist()
            }
        final_results[str(g)] = summary

    # Output
    tag = f"gamma_ablation_{args.env}"
    out_dir = Path(args.out_dir).resolve()
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump({"K": args.K, "seeds": args.seeds, "gammas": gammas, "results": final_results}, f, indent=2)
    
    plot_path = Path(args.plot_dir).resolve() / f"{tag}.png"
    plot_ablation(final_results, gammas, args.K, args.seeds, args.n_steps, env_label, str(plot_path))
    
    print(f"\nResults: {out_dir / f'{tag}.json'}")
    print(f"Plot:    {plot_path}")

if __name__ == "__main__":
    main()

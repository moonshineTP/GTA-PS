"""
RQ2 Ablation: Beta Magnitude vs. Proactive Correlation.

Claims:
1. Beta magnitude controls the "proactive intensity" (dominance of topology signal).
2. This is measured via Spearman rho between the topology field and actual pop order.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    # Try reaching up to find the real project root (e.g. if compiled in artifacts/dist/)
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.envs.garnet import GarnetEnv
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms import make_agent
from runner import RunManager
from plot_academic import apply_style, get_palette, get_utility_color

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def run_beta_sweep(
    env_name: str,
    n_states: int,
    seed: int,
    gamma: float,
    beta_vals: list,
    K: int,
    k_fourrooms: int | None = None,
    **env_kwargs
) -> dict:
    if env_name == "garnet":
        env = GarnetEnv(n_states=n_states, gamma=gamma, seed=seed, **env_kwargs)
    else:
        env = FourRoomsTabular(k=k_fourrooms, gamma=gamma, seed=seed)
        n_states = env.n_states

    # Initial reward shift to set up tau
    r = env.shift_reward()
    
    rhos = []
    for beta in beta_vals:
        # Create GTA-PS agent with zero initial reward to force proactive signal on update
        agent = make_agent(
            "gta_base", host="dp", P=env.P, r=np.zeros(n_states), gamma=gamma,
            beta=beta, alpha=0.5
        )
        
        # Apply reward shift from zero to r
        agent.module.reward_update(r)
        
        # Record initial topology ranking
        tau = agent.module.tau.copy()
        
        # Capture pop sequence
        sequence = []
        for _ in range(K):
            s = agent.module.pop_and_prioritize()
            if s is None: break
            sequence.append(s)
            delta_V = agent.module.apply_bellman_backup(s, agent.module._P_model, agent.module._r_model, agent.gamma, agent.module.V)
            agent.module.notify_backup(s, np.zeros(agent.module.n_actions), delta_V)
            agent.module.push_predecessors(s, delta_V)
        
        # Calculate rho
        if not sequence or np.all(tau == tau[0]) or np.max(np.abs(tau)) < 1e-12:
            rho = 0.0
        else:
            initial_tau_ranks = stats.rankdata(-tau, method='min')
            pop_order = np.arange(len(sequence)) + 1
            tau_ranks = [initial_tau_ranks[s] for s in sequence]
            rho_raw = stats.spearmanr(pop_order, tau_ranks)[0]
            if isinstance(rho_raw, tuple):
                rho = 0.0
            else:
                rho = float(str(rho_raw))
                if np.isnan(rho):
                    rho = 0.0
        
        if seed == 0:
            print(f"DEBUG: beta={beta:6.3f} tau_max={np.max(np.abs(tau)):.6f} sequence_len={len(sequence)} rho={rho:.4f}")
            
        rhos.append(rho)
        
    return {"rhos": rhos}


def plot_rq2_results(results_dir: Path, plot_dir: Path) -> Path | None:
    if plt is None:
        return None

    apply_style()
    try:
        with open(results_dir / "rq2_ablation_garnet_s75.json") as f:
            garnet_data = json.load(f)
        with open(results_dir / "rq2_ablation_fourrooms_k4.json") as f:
            fr_data = json.load(f)
    except FileNotFoundError:
        return None

    point_colors = get_palette("point")
    colors = {
        "garnet": point_colors["standard_ps"],
        "fourrooms": point_colors["gta_base"],
    }
    grid_color = get_utility_color("grid")

    fig, ax = plt.subplots(figsize=(8, 5))
    beta_vals = garnet_data["beta_vals"]
    ax.errorbar(
        beta_vals,
        garnet_data["mean_rhos"],
        yerr=garnet_data["sem_rhos"],
        label="GARNET",
        marker="o",
        capsize=5,
        color=colors["garnet"],
    )
    ax.errorbar(
        beta_vals,
        fr_data["mean_rhos"],
        yerr=fr_data["sem_rhos"],
        label="FourRooms",
        marker="s",
        capsize=5,
        color=colors["fourrooms"],
    )
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlabel(r"$\beta$ Magnitude")
    ax.set_ylabel("Correlation")
    ax.set_title("Beta Allocation Sweep")
    ax.legend()
    ax.grid(True, color=grid_color)
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "rq2_beta_ablation.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="garnet", choices=["garnet", "fourrooms"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--plot-dir", default=str(ROOT / "plots"))
    args = parser.parse_args()

    beta_vals = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    if args.env == "garnet":
        env_kwargs = {"n_states": 75, "n_actions": 10, "branching": 3, "reward_support": 4}
        K = 25 # budget
        tag = "garnet_s75"
    else:
        env_kwargs = {"n_states": 68, "k_fourrooms": 4}
        K = 25
        tag = "fourrooms_k4"

    runner = RunManager(n_workers=args.workers)
    fn_kwargs = {
        "env_name": args.env,
        "beta_vals": beta_vals,
        "K": K,
        "gamma": 0.95,
        **env_kwargs
    }
    
    print(f"Running beta ablation on {args.env} ({args.seeds} seeds)...")
    results = runner.run(run_beta_sweep, list(range(args.seeds)), fn_kwargs)
    
    # Aggregate
    all_rhos = np.array([r["rhos"] for r in results])
    mean_rhos = np.mean(all_rhos, axis=0)
    std_rhos = np.std(all_rhos, axis=0) / np.sqrt(args.seeds) # SEM
    
    print(f"\nResults for {args.env}:")
    print(f"{'Beta':>10} | {'Mean Rho':>10} | {'SEM':>10}")
    print("-" * 35)
    for i, b in enumerate(beta_vals):
        print(f"{b:10.3f} | {mean_rhos[i]:10.4f} | {std_rhos[i]:10.4f}")

    # Output to JSON
    payload = {
        "experiment": "rq2_beta_ablation",
        "env": args.env,
        "beta_vals": beta_vals,
        "mean_rhos": mean_rhos.tolist(),
        "sem_rhos": std_rhos.tolist()
    }
    out_dir = Path(args.out_dir).resolve()
    out_path = (out_dir / f"rq2_ablation_{tag}.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    plot_path = plot_rq2_results(out_dir, Path(args.plot_dir).resolve())
    print(f"\nResults saved to: {out_path}")
    if plot_path is not None:
        print(f"Plot saved to:    {plot_path}")
    else:
        print("Plot skipped:     waiting for both rq2_ablation_garnet_s75.json and rq2_ablation_fourrooms_k4.json")

if __name__ == "__main__":
    main()

"""
RQ1: Convergence rate and trajectory in stationary environments.

This script measures the value error trajectory (L1 and L-inf norms) as a function of
host steps (each performing K backups) under a NO reward shift scenario.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.envs.garnet import GarnetEnv
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.algorithms import make_agent
from gta_ps.utils.metrics import value_error_inf, value_error_l1
from runner import RunManager
from plot_academic import (
    LABELS as AGENT_LABELS,
    apply_style,
    get_palette,
    get_publication_defaults,
    get_utility_color,
    warn_if_nonpublication_run,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

PUBLICATION_DEFAULTS = get_publication_defaults()

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_trajectory_seed(
    env_name: str,
    n_states: int | None,
    k_fourrooms: int | None,
    reward_support: int | None,
    seed: int,
    gamma: float,
    n_steps: int,
    K: int,
    method_cfgs: dict,
    n_actions: int = 10,
    branching: int = 3,
    n_nbr_rate: float = 0.15,
    arnoldi_k: int = 15,
    **ignored,
) -> dict:
    """Run convergence trajectory for one seed (no shifts)."""
    if env_name == "garnet":
        env = GarnetEnv(n_states=n_states, n_actions=n_actions, branching=branching,
                        reward_support=reward_support, n_nbr_rate=n_nbr_rate,
                        gamma=gamma, seed=seed)
    else:
        env = FourRoomsTabular(k=k_fourrooms, gamma=gamma, seed=seed)
        n_states = env.n_states

    variants = ["standard_ps", "gta_base", "gta_beta"]
    r = env.r
    V_star = env.value_iteration(r)
    
    agents = {}
    for v in variants:
        kw = {"arnoldi_k": arnoldi_k}
        kw.update(method_cfgs.get(v, {}))
        agents[v] = make_agent(v, host="dp", P=env.P, r=r, gamma=gamma, q=20, **kw)
    
    traces = {v: {"inf": [], "l1": []} for v in variants}
    for _ in range(n_steps):
        for v in variants:
            V = agents[v].plan(K=K)
            traces[v]["inf"].append(float(value_error_inf(V, V_star)))
            traces[v]["l1"].append(float(value_error_l1(V, V_star)))

    return traces

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot_trajectory(results, K, seeds, n_steps, env_label, out_path):
    if plt is None: return
    apply_style()
    colors = get_palette("line")
    fallback_color = get_utility_color("text_main")
    grid_color = get_utility_color("grid")
    steps_arr = np.arange(n_steps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    norm_cfg = [
        ("l1",  r"Value Error ($L_1$, mean abs)"),
        ("inf", r"Value Error ($L_\infty$, max abs)"),
    ]
    for ax, (norm, ylabel) in zip(axes, norm_cfg):
        for v in results:
            mean = np.array(results[v][norm]["mean"])
            ci   = np.array(results[v][norm]["ci95"])
            label = AGENT_LABELS.get(v, v)
            color = colors.get(v, fallback_color)
            ax.plot(steps_arr, mean, label=label, color=color, linewidth=2)
            ax.fill_between(steps_arr, mean - ci, mean + ci, color=color, alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel(f"Host Step (K={K} backups/step)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls="-", color=grid_color, alpha=0.2)
        ax.legend()

    fig.suptitle(f"Convergence Trajectory: {env_label}", y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ1: Convergence Trajectory in Stationary Environments")
    parser.add_argument("--env", default="garnet", choices=["garnet", "fourrooms"])
    parser.add_argument("--n-states", type=int, default=75, help="GARNET only: number of states")
    parser.add_argument("--k-fourrooms", type=int, default=1, help="FourRooms only: config k")
    parser.add_argument("--K", type=int, default=10, help="Planning budget per step")
    parser.add_argument("--n-steps", type=int, default=150, help="Number of host steps to record")
    parser.add_argument("--seeds", type=int, default=PUBLICATION_DEFAULTS["seeds"])
    parser.add_argument("--workers", default=PUBLICATION_DEFAULTS["workers"], help="Number of workers or 'auto'")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--plot-dir", default=str(ROOT / "plots"))

    args = parser.parse_args()
    warn_if_nonpublication_run(args.seeds, args.workers)
    
    # Load tuned hyperparameters
    from gta_ps.utils.hyperparams import load_tuned_for_env
    method_cfgs = load_tuned_for_env(args.env)
    
    env_label = f"Garnet(S={args.n_states})" if args.env == "garnet" else f"FourRooms(k={args.k_fourrooms})"
    file_tag = f"{args.env}_S{args.n_states if args.env=='garnet' else 'FR'}"
    
    print(f"\n[Convergence] Mission: {env_label}  K={args.K}  seeds={args.seeds}  steps={args.n_steps}")
    
    runner = RunManager(n_workers=args.workers)
    fn_kwargs = dict(
        env_name=args.env, n_states=args.n_states, k_fourrooms=args.k_fourrooms,
        reward_support=4, gamma=0.95, n_steps=args.n_steps, K=args.K,
        method_cfgs=method_cfgs
    )
    
    all_results = runner.run(run_trajectory_seed, range(args.seeds), fn_kwargs)
    
    # Aggregate stats
    from scipy import stats
    variants = list(all_results[0].keys())
    summary = {}
    for v in variants:
        summary[v] = {}
        for norm in ("inf", "l1"):
            mat = np.array([r[v][norm] for r in all_results])
            summary[v][norm] = {
                "mean": mat.mean(axis=0).tolist(),
                "ci95": (stats.sem(mat, axis=0) * stats.t.ppf(0.975, df=len(mat)-1)).tolist()
            }
    
    # Output
    out_dir = Path(args.out_dir).resolve()
    json_path = out_dir / f"rq1_proactiveness_{file_tag}_K{args.K}.json"
    with open(json_path, "w") as f:
        json.dump({"K": args.K, "seeds": args.seeds, "results": summary}, f, indent=2)
    
    plot_path = Path(args.plot_dir).resolve() / f"rq1_proactiveness_{file_tag}_K{args.K}.png"
    plot_trajectory(summary, args.K, args.seeds, args.n_steps, env_label, str(plot_path))
    
    print(f"Results saved to: {json_path}")
    print(f"Plot saved to:    {plot_path}\n")

if __name__ == "__main__":
    main()

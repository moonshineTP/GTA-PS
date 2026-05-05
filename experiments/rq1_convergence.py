"""
RQ1: Stationary convergence across environments and planning budgets.

This script measures value-error trajectories without reward shifts and compares
two environments (GARNET and FourRooms) under two planning budgets by default.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():
    for p in Path(__file__).resolve().parents:
        if (p / "configs").exists() and (p / "src").exists():
            ROOT = p
            break
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))
sys.path.insert(0, str(ROOT / "misc"))

from gta_ps.algorithms import make_agent
from gta_ps.envs.fourrooms import FourRoomsTabular
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.utils.hyperparams import load_tuned_for_env
from gta_ps.utils.metrics import value_error_inf, value_error_l1
from plot_academic import LABELS as AGENT_LABELS, apply_style, get_palette, get_utility_color
from runner import RunManager

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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
    n_actions: int = 4,
    branching: int = 3,
    n_nbr_rate: float = 0.15,
    arnoldi_k: int = 15,
    **ignored,
) -> dict:
    """Run one stationary convergence trajectory for one seed."""
    if env_name == "garnet":
        env = GarnetEnv(
            n_states=n_states,
            n_actions=n_actions,
            branching=branching,
            reward_support=reward_support,
            n_nbr_rate=n_nbr_rate,
            gamma=gamma,
            seed=seed,
        )
    else:
        env = FourRoomsTabular(k=k_fourrooms, gamma=gamma, seed=seed)

    variants = ["standard_ps", "gta_base", "gta_beta"]
    V_star = env.value_iteration(env.r)
    agents = {}
    for variant in variants:
        kwargs = {"arnoldi_k": arnoldi_k}
        kwargs.update(method_cfgs.get(variant, {}))
        agents[variant] = make_agent(
            variant,
            host="dp",
            P=env.P,
            r=env.r,
            gamma=gamma,
            q=20,
            **kwargs,
        )

    traces = {variant: {"l1": [], "inf": []} for variant in variants}
    for _ in range(n_steps):
        for variant in variants:
            V = agents[variant].plan(K=K)
            traces[variant]["l1"].append(float(value_error_l1(V, V_star)))
            traces[variant]["inf"].append(float(value_error_inf(V, V_star)))
    return traces


def aggregate_trajectories(all_results: list[dict]) -> dict:
    variants = list(all_results[0].keys())
    summary = {}
    for variant in variants:
        summary[variant] = {}
        for norm in ("l1", "inf"):
            mat = np.array([result[variant][norm] for result in all_results], dtype=float)
            ci = np.zeros(mat.shape[1], dtype=float)
            if mat.shape[0] > 1:
                ci = stats.sem(mat, axis=0) * stats.t.ppf(0.975, df=mat.shape[0] - 1)
            summary[variant][norm] = {
                "mean": mat.mean(axis=0).tolist(),
                "ci95": ci.tolist(),
            }
    return summary


def plot_convergence(summary_by_env: dict, budgets: list[int], n_steps: int, out_path: Path) -> bool:
    if plt is None:
        return False

    apply_style()
    colors = get_palette("line")
    fallback_color = get_utility_color("text_main")
    grid_color = get_utility_color("grid")
    steps = np.arange(n_steps)
    line_styles = ["-", "--", "-.", ":"]
    style_map = {K: line_styles[idx % len(line_styles)] for idx, K in enumerate(budgets)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    env_order = ["garnet", "fourrooms"]

    for ax, env_name in zip(axes, env_order):
        env_payload = summary_by_env[env_name]
        for variant in ("standard_ps", "gta_base", "gta_beta"):
            color = colors.get(variant, fallback_color)
            for K in budgets:
                run = env_payload["budgets"][K]
                mean = np.array(run[variant]["l1"]["mean"], dtype=float)
                ci = np.array(run[variant]["l1"]["ci95"], dtype=float)
                label = f"{AGENT_LABELS.get(variant, variant)} (K={K})"
                ax.plot(
                    steps,
                    mean,
                    color=color,
                    linestyle=style_map[K],
                    linewidth=2,
                    label=label,
                )
                ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.10)
        ax.set_title(env_payload["label"])
        ax.set_xlabel("Host Step")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", color=grid_color, alpha=0.2)

    axes[0].set_ylabel(r"Value Error ($L_1$)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("RQ1 Convergence in Stationary Environments", y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="RQ1: Stationary convergence across environments and budgets")
    parser.add_argument("--n-states", type=int, default=75, help="GARNET only: number of states")
    parser.add_argument("--reward-support", type=int, default=3, help="GARNET only: reward support size")
    parser.add_argument("--k-fourrooms", type=int, default=1, help="FourRooms k")
    parser.add_argument("--n-actions", type=int, default=4, help="GARNET only: number of actions")
    parser.add_argument("--branching", type=int, default=3, help="GARNET only: branching factor")
    parser.add_argument("--n-nbr-rate", type=float, default=0.15, help="GARNET only: neighborhood rate")
    parser.add_argument("--budgets", nargs="+", type=int, default=[5, 20], help="Planning budgets per host step")
    parser.add_argument("--n-steps", type=int, default=150, help="Number of host steps to record")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", default=5, help="Number of workers or 'auto'")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--plot-dir", default=str(ROOT / "plots"))
    args = parser.parse_args()

    budgets = [int(K) for K in args.budgets]
    runner = RunManager(n_workers=args.workers)
    out_dir = Path(args.out_dir).resolve()
    plot_dir = Path(args.plot_dir).resolve()

    env_specs = {
        "garnet": {
            "label": f"GARNET (|S|={args.n_states}, supp={args.reward_support})",
            "kwargs": {
                "env_name": "garnet",
                "n_states": args.n_states,
                "k_fourrooms": None,
                "reward_support": args.reward_support,
                "n_actions": args.n_actions,
                "branching": args.branching,
                "n_nbr_rate": args.n_nbr_rate,
            },
        },
        "fourrooms": {
            "label": f"FourRooms (k={args.k_fourrooms})",
            "kwargs": {
                "env_name": "fourrooms",
                "n_states": None,
                "k_fourrooms": args.k_fourrooms,
                "reward_support": None,
            },
        },
    }

    summary_by_env: dict = {}
    for env_name, env_spec in env_specs.items():
        method_cfgs = load_tuned_for_env(env_name)
        summary_by_env[env_name] = {"label": env_spec["label"], "budgets": {}}
        for K in budgets:
            print(f"\n[Convergence] env={env_name}  K={K}  seeds={args.seeds}  steps={args.n_steps}")
            fn_kwargs = {
                **env_spec["kwargs"],
                "gamma": args.gamma,
                "n_steps": args.n_steps,
                "K": K,
                "method_cfgs": method_cfgs,
            }
            all_results = runner.run(run_trajectory_seed, list(range(args.seeds)), fn_kwargs)
            summary_by_env[env_name]["budgets"][K] = aggregate_trajectories(all_results)

    payload = {
        "experiment": "rq1_convergence",
        "seeds": args.seeds,
        "n_steps": args.n_steps,
        "budgets": budgets,
        "envs": {
            env_name: {
                "label": env_payload["label"],
                "budgets": {str(K): env_payload["budgets"][K] for K in budgets},
            }
            for env_name, env_payload in summary_by_env.items()
        },
    }

    json_path = out_dir / f"rq1_convergence_garnetS{args.n_states}_fourroomsK{args.k_fourrooms}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    plot_path = plot_dir / f"rq1_convergence_garnetS{args.n_states}_fourroomsK{args.k_fourrooms}.png"
    plot_written = plot_convergence(summary_by_env, budgets, args.n_steps, plot_path)

    for env_name, env_payload in summary_by_env.items():
        print(f"\nEnvironment: {env_payload['label']}")
        for K in budgets:
            print(f"  K={K}")
            for variant in ("standard_ps", "gta_base", "gta_beta"):
                final_l1 = env_payload["budgets"][K][variant]["l1"]["mean"][-1]
                final_inf = env_payload["budgets"][K][variant]["inf"]["mean"][-1]
                print(f"    {variant:12s} final L1={final_l1:.4f}  final Linf={final_inf:.4f}")

    print(f"\nResults saved to: {json_path}")
    if plot_written:
        print(f"Plot saved to:    {plot_path}")
    else:
        print("Plot skipped:     matplotlib is not installed")


if __name__ == "__main__":
    main()

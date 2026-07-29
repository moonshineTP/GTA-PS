"""Swift GTA-beta stability pass for the dense-support GARNET-200 protocol."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "threads"))

from gta_ps.algorithms import make_agent
from gta_ps.envs.garnet import GarnetEnv
from gta_ps.utils.hyperparams import load_tuned_for_env
from runner import RunManager
from tuner import TuningEngine

from wall_clock_multishift import sparse_value_iteration


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative floats")
    return values


def _score_config(
    *,
    seed: int,
    beta1: float,
    eta: float,
    n_states: int,
    n_actions: int,
    branching: int,
    n_nbr_rate: float,
    reward_support: int,
    gamma: float,
    n_shifts: int,
    backups_per_shift: int,
    fixed_gta_config: dict[str, Any],
) -> float:
    env = GarnetEnv(
        n_states=n_states,
        n_actions=n_actions,
        branching=branching,
        reward_support=reward_support,
        gamma=gamma,
        n_nbr_rate=n_nbr_rate,
        reward_locality=True,
        seed=seed,
    )
    initial_reward = env.r.copy()
    config = {**fixed_gta_config, "beta1": beta1, "eta": eta}
    agent = make_agent(
        "gta_beta",
        host="dp",
        P=env.P,
        r=initial_reward,
        gamma=gamma,
        q=None,
        **config,
    )

    errors = []
    for _ in range(n_shifts):
        reward = env.shift_reward()
        target = sparse_value_iteration(
            env.P,
            reward,
            gamma,
            branching=branching,
        )
        value = agent.plan(K=backups_per_shift, r=reward)
        errors.append(float(np.mean(np.abs(value - target))))
    return float(np.mean(errors))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune beta1 and eta for stable GARNET-200 replanning."
    )
    parser.add_argument("--beta1-values", type=_parse_float_list, default="0,0.01,0.025,0.05,0.1")
    parser.add_argument("--eta-values", type=_parse_float_list, default="0.0526315789,0.1,0.2")
    parser.add_argument("--n-states", type=int, default=200)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--n-nbr-rate", type=float, default=0.10)
    parser.add_argument("--reward-support", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-shifts", type=int, default=1)
    parser.add_argument("--backups-per-shift", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "results"
        / "rebuttal"
        / "wall_clock_multishift_garnet200"
        / "tuning.json",
    )
    args = parser.parse_args()

    inherited = dict(load_tuned_for_env("garnet_s200_dense").get("gta_beta") or {})
    fixed = {
        key: inherited[key]
        for key in ("alpha", "beta2", "b", "slem_recompute_every")
        if key in inherited
    }
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    engine = TuningEngine(RunManager(n_workers=args.workers))
    best, records = engine.search(
        _score_config,
        param_grid={
            "beta1": args.beta1_values,
            "eta": args.eta_values,
        },
        fixed_kwargs={
            "n_states": args.n_states,
            "n_actions": args.n_actions,
            "branching": args.branching,
            "n_nbr_rate": args.n_nbr_rate,
            "reward_support": args.reward_support,
            "gamma": args.gamma,
            "n_shifts": args.n_shifts,
            "backups_per_shift": args.backups_per_shift,
            "fixed_gta_config": fixed,
        },
        seeds=seeds,
        lower_is_better=True,
        desc="garnet200-beta-eta",
    )
    beta2 = float(fixed.get("beta2", 0.0))
    for record in records:
        params = record["params"]
        record["sufficient_condition_margin"] = float(
            params["eta"] - params["beta1"] * (1.0 + beta2)
        )
    nonzero = [record for record in records if record["params"]["beta1"] > 0.0]
    sufficient = [
        record
        for record in nonzero
        if record["sufficient_condition_margin"] > 0.0
    ]
    regularized = [
        record for record in sufficient if record["params"]["eta"] >= 0.1
    ]
    recommended = regularized[0] if regularized else (sufficient[0] if sufficient else records[0])
    output = {
        "protocol": {
            "environment": "GARNET",
            "n_states": args.n_states,
            "n_actions": args.n_actions,
            "branching": args.branching,
            "n_nbr_rate": args.n_nbr_rate,
            "reward_support": args.reward_support,
            "gamma": args.gamma,
            "n_shifts": args.n_shifts,
            "backups_per_shift": args.backups_per_shift,
            "seeds": seeds,
            "workers": args.workers,
            "objective": "mean post-shift L1 value error under a fixed backup budget",
        },
        "fixed_gta_config": fixed,
        "best": best,
        "best_nonzero": nonzero[0]["params"] if nonzero else None,
        "best_sufficient_nonzero": sufficient[0]["params"] if sufficient else None,
        "recommended": recommended["params"],
        "recommendation_rule": (
            "lowest mean error among nonzero candidates satisfying "
            "beta1*(1+beta2)<eta and eta>=0.1"
        ),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"best={best}")
    print(f"best_nonzero={output['best_nonzero']}")
    print(f"recommended={output['recommended']}")
    print(f"results={args.out}")


if __name__ == "__main__":
    main()

"""
RQ2: Topology allocation under adaptive schedules.

Does the adaptive GTA-PS^beta schedule match or outperform the best fixed
allocation once geometry varies across shifts?

Protocol (memo §8):
  - GARNET |S|=75/200 (generative variety)
  - Compare fixed beta (gta_base) vs adaptive beta_t (gta_beta)
  - Track SLEM_t and beta_t
  - Metric: Spearman rank correlation between tau and actual pop order.
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
from gta_ps.utils.metrics import value_error_inf, value_error_l1
from runner import RunManager
from plot_academic import apply_style, get_palette

# ---------------------------------------------------------------------------
# Config & Factory
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required for --config: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _apply_yaml_defaults(args: argparse.Namespace, cfg: dict) -> None:
    _MAP = [
        ("env",            "env",            "garnet"),
        ("n_states",       "n_states",        None),
        ("k_fourrooms",    "k_fourrooms",     None),
        ("reward_support", "reward_support",  None),
        ("gamma",          "gamma",           0.95),
        ("n_shifts",       "n_shifts",        5),
        ("seeds",          "seeds",           30),
    ]
    for dest, key, default in _MAP:
        if key in cfg and getattr(args, dest) == default:
            setattr(args, dest, cfg[key])
    if "budgets" in cfg:
        args.yaml_budgets = cfg["budgets"]

def _build_method_cfgs(cfg: dict) -> dict:
    root_defaults = {
        k: cfg.get(k) for k in ["alpha", "beta", "beta1", "beta2", "eta", "b"]
        if cfg.get(k) is not None
    }
    default_block = cfg.get("default") or {}
    base_params = {**root_defaults, **default_block}
    
    tuned = cfg.get("tuned") or {}
    result  = {}
    for variant in ("gta_base", "gta_beta"):
        variant_cfg = {k: v for k, v in base_params.items() if v is not None}
        overrides = {k: v for k, v in (tuned.get(variant) or {}).items() if v is not None}
        result[variant] = {**variant_cfg, **overrides}
    return result

def resolve_run_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    if args.env == "garnet":
        n_states = args.n_states if args.n_states is not None else 75
        reward_support = args.reward_support if args.reward_support is not None else 4
        budgets = list(args.yaml_budgets) if getattr(args, "yaml_budgets", None) else [n_states]
        return {
            "env_name": "garnet", "n_states": n_states, "k_fourrooms": None,
            "reward_support": reward_support, "budgets": budgets,
            "env_label": f"garnet (|S|={n_states}, supp={reward_support})",
            "file_tag": f"garnet_S{n_states}_supp{reward_support}",
        }
    else:
        k_fr = args.k_fourrooms if args.k_fourrooms is not None else 1
        env = FourRoomsTabular(k=k_fr, gamma=args.gamma, seed=0)
        budgets = list(args.yaml_budgets) if getattr(args, "yaml_budgets", None) else [env.n_states]
        return {
            "env_name": "fourrooms", "n_states": env.n_states, "k_fourrooms": k_fr,
            "reward_support": None, "budgets": budgets,
            "env_label": f"fourrooms (k={k_fr}, |S|={env.n_states})",
            "file_tag": f"fourrooms_k{k_fr}_S{env.n_states}",
        }

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def run_single_seed(
    env_name: str,
    n_states: int | None,
    k_fourrooms: int | None,
    reward_support: int | None,
    seed: int,
    gamma: float,
    n_shifts: int,
    n_actions: int,
    branching: int,
    n_nbr_rate: float,
    arnoldi_k: int = 15,
    method_cfgs: dict | None = None,
    budgets: list | None = None,
    **ignored,
) -> dict:
    if env_name == "garnet":
        env = GarnetEnv(n_states=n_states, n_actions=n_actions, branching=branching,
                        reward_support=reward_support, n_nbr_rate=n_nbr_rate,
                        gamma=gamma, seed=seed)
    else:
        env = FourRoomsTabular(k=k_fourrooms, gamma=gamma, seed=seed)
        n_states = env.n_states

    if budgets is None:
        budgets = [n_states]

    _mc = method_cfgs or {}
    def _module_kwargs(variant: str) -> dict:
        kw = {"arnoldi_k": arnoldi_k}
        kw.update(_mc.get(variant, {}))
        return kw

    variants = ["standard_ps", "gta_base", "gta_beta"]
    results_inf: dict = {v: {K: [] for K in budgets} for v in variants}
    results_l1:  dict = {v: {K: [] for K in budgets} for v in variants}
    history: dict = {v: [] for v in variants if v.startswith("gta")}
    
    # Pre-build agents
    agents: dict = {}
    for K in budgets:
        for v in variants:
            agents[(K, v)] = make_agent(
                v, host="dp", P=env.P, r=env.r, gamma=gamma, q=20,
                **_module_kwargs(v),
            )

    # Instrumentation for rank correlation
    def get_order_correlation(agent, K, r):
        # 1. Record initial topology ranking AFTER reward shift update
        tau = agent.module.tau.copy()
        
        # 2. Perform planning and record sequence
        sequence = []
        # Store state to restore later
        V_orig = agent.module.V.copy()
        xi_orig = agent.module.xi.copy()
        
        # We need to manually duplicate the plan() logic to capture sequence
        for _ in range(K):
            s = agent.module.pop_and_prioritize()
            if s is None: break
            sequence.append(s)
            delta_V = agent.module.apply_bellman_backup(s, agent.module._P_model, agent.module._r_model, agent.gamma, agent.module.V)
            ev_s = (agent.module.P[:, s, :] * agent.module.V[None, :]).sum(axis=1)
            q_s = agent.module._r_model[s] + agent.gamma * ev_s
            adv_PS_row = q_s - q_s.max()
            agent.module.notify_backup(s, adv_PS_row, delta_V)
            agent.module.push_predecessors(s, delta_V)
            
        # Restore agent state so it's ready for the next shift
        agent.module.V = V_orig
        agent.module.xi = xi_orig
        agent.module._refresh_top_q()
        agent.module._rekey_residual_queue()
            
        if not sequence: return 0.0
        
        # 3. Correlation: sequence rank vs rank in initial tau
        if np.all(tau == tau[0]) or np.max(np.abs(tau)) < 1e-12:
            return 0.0
            
        initial_tau_ranks = stats.rankdata(-tau, method='min')
        
        # Spearman correlation between sequence order and their initial tau ranks
        if len(sequence) < 2: return 1.0
        
        # Compare actual pop order (1..len(seq)) with their topology ranks
        pop_order = np.arange(len(sequence)) + 1
        tau_ranks = [initial_tau_ranks[s] for s in sequence]
        
        rho_raw = stats.spearmanr(pop_order, tau_ranks)[0]
        if isinstance(rho_raw, tuple):
            return 0.0

        rho = float(str(rho_raw))
        return 0.0 if np.isnan(rho) else rho

    for _ in range(n_shifts):
        r = env.shift_reward()
        V_star = env.value_iteration(r)
        
        for K in budgets:
            for v in variants:
                agent = agents[(K, v)]
                # agent.r is current agent reward. shift_reward gives new r.
                delta_r = r - agent.module.r
                agent.module.reward_update(delta_r)
                
                if v.startswith("gta") and K == budgets[0]:
                    rho = get_order_correlation(agent, K, r)
                    history[v].append({
                        "slem": float(agent.module.slem),
                        "beta_t": float(agent.module.beta_t),
                        "rho": rho
                    })
                    # Re-run plan for actual value update (get_order_correlation restored V)
                    V = agent.plan(K=K, r=None) 
                else:
                    V = agent.plan(K=K, r=None) # r already applied via reward_update
                
                results_inf[v][K].append(value_error_inf(V, V_star))
                results_l1[v][K].append(value_error_l1(V, V_star))

    return {"results": {"inf": results_inf, "l1": results_l1}, "history": history}

def aggregate(all_results: list, budgets: list) -> dict:
    methods = list(all_results[0]["results"]["inf"].keys())
    out = {"results": {}, "history": {}}

    for norm in ("inf", "l1"):
        out["results"][norm] = {}
        for m in methods:
            out["results"][norm][m] = {}
            for K in budgets:
                vals = np.array([r["results"][norm][m][K] for r in all_results]).ravel()
                mean = round(float(np.mean(vals)), 6)
                ci   = round(float(stats.sem(vals) * stats.t.ppf(0.975, df=len(vals) - 1)), 6)
                out["results"][norm][m][K] = {"mean": mean, "ci95": ci, "n": len(vals)}
            
    for m in ["gta_base", "gta_beta"]:
        slems, betas, rhos = [], [], []
        for r in all_results:
            slems.append([h["slem"] for h in r["history"][m]])
            betas.append([h["beta_t"] for h in r["history"][m]])
            rhos.append([h["rho"] for h in r["history"][m]])
        out["history"][m] = {
            "slem_mean": np.mean(slems, axis=0).tolist(),
            "beta_mean": np.mean(betas, axis=0).tolist(),
            "rho_mean": np.mean(rhos, axis=0).tolist(),
        }
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", default="garnet", choices=["garnet", "fourrooms"])
    parser.add_argument("--n-states", type=int, help="GARNET only: number of states")
    parser.add_argument("--k-fourrooms", type=int, help="FourRooms only: config k")
    parser.add_argument("--reward-support", type=int, choices=[1, 2, 4, 8, 16], help="GARNET only: |supp(r)|")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n-shifts", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))

    args = parser.parse_args()
    _cfg = _load_config(str(Path(args.config).resolve())) if args.config else {}
    _apply_yaml_defaults(args, _cfg)
    run_config = resolve_run_config(args, parser)
    budgets = run_config["budgets"]

    if args.config:
        _method_cfgs = _build_method_cfgs(_cfg)
    else:
        from gta_ps.utils.hyperparams import load_tuned_for_env
        _method_cfgs = load_tuned_for_env(run_config["env_name"])

    fixed_env_kwargs = dict(
        env_name=run_config["env_name"], n_states=run_config["n_states"],
        k_fourrooms=run_config["k_fourrooms"], reward_support=run_config["reward_support"],
        gamma=args.gamma, n_shifts=args.n_shifts, 
        n_actions=_cfg.get("n_actions", 10), branching=_cfg.get("branching", 3),
        n_nbr_rate=_cfg.get("n_nbr_rate", 0.15), arnoldi_k=_cfg.get("arnoldi_k", 15),
        method_cfgs=_method_cfgs, budgets=budgets,
    )

    runner = RunManager(n_workers=args.workers)
    eval_seeds = [s for s in range(args.seeds)]
    print(f"\n[eval] {len(eval_seeds)} seeds  workers={args.workers}")
    
    all_results = runner.run(run_single_seed, eval_seeds, fixed_env_kwargs)
    summary = aggregate(all_results, budgets)

    payload = {
        "experiment": "rq2_allocation",
        "env": run_config["env_label"],
        "results": summary,
    }
    out_dir = Path(args.out_dir).resolve()
    out_path = out_dir / f"rq2_{run_config['file_tag']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nEnvironment: {run_config['env_label']}")
    for norm in ("l1", "inf"):
        print(f"  --- L_{norm} ---")
        for method, kdict in summary["results"][norm].items():
            row = f"  {method:12s}"
            for K, st in kdict.items():
                row += f"  K={K}: {st['mean']:.4f}±{st['ci95']:.4f}"
            print(row)
    
    for m in ["gta_base", "gta_beta"]:
        rho = summary["history"][m]["rho_mean"]
        print(f"  {m:10s} avg rho: {np.mean(rho):.4f}")

if __name__ == "__main__":
    main()

"""Aggregate and plot the GARNET-200 Arnoldi cadence ablation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "misc"))

from plot_academic import apply_style, get_palette, get_utility_color  # noqa: E402

CADENCES = (15, 200, 1_000_000)
LABELS = ("15", "200", "Initialization only")


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        ci95 = 0.0
    else:
        ci95 = float(
            stats.t.ppf(0.975, values.size - 1)
            * values.std(ddof=1)
            / np.sqrt(values.size)
        )
    return {"mean": float(values.mean()), "ci95": ci95}


def aggregate(input_dir: Path) -> dict[str, Any]:
    conditions = []
    condition_arrays = []
    expected_seeds: list[int] | None = None
    for cadence, label in zip(CADENCES, LABELS):
        path = input_dir / f"period_{cadence}" / "wall_clock_multishift_time.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        seeds = [int(row["seed"]) for row in payload["raw"]]
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError("cadence conditions must use identical ordered seeds")

        gta_rows = [row["records"]["gta_beta"] for row in payload["raw"]]
        standard_rows = [row["records"]["standard_ps"] for row in payload["raw"]]
        gta_rates = np.asarray(
            [row["mean_completed_backups_per_second"] for row in gta_rows]
        )
        standard_rates = np.asarray(
            [row["mean_completed_backups_per_second"] for row in standard_rows]
        )
        conditions.append(
            {
                "cadence_backups": cadence,
                "label": label,
                "slem_refreshes_per_period": _summary(
                    np.asarray(
                        [row["mean_slem_refreshes_per_period"] for row in gta_rows]
                    )
                ),
                "gta_backups_per_second": _summary(gta_rates),
                "standard_backups_per_second": _summary(standard_rates),
                "relative_throughput": _summary(gta_rates / standard_rates),
                "gta_l1_recovery_auc": _summary(
                    np.asarray([row["recovery_auc"] for row in gta_rows])
                ),
                "gta_linf_recovery_auc": _summary(
                    np.asarray([row["recovery_auc_linf"] for row in gta_rows])
                ),
            }
        )
        condition_arrays.append(
            {
                "gta_rate": gta_rates,
                "relative_throughput": gta_rates / standard_rates,
                "l1_auc": np.asarray([row["recovery_auc"] for row in gta_rows]),
                "linf_auc": np.asarray(
                    [row["recovery_auc_linf"] for row in gta_rows]
                ),
            }
        )

    comparisons = {}
    for name, current, reference in (
        ("period_200_minus_period_15", 1, 0),
        ("initialization_only_minus_period_200", 2, 1),
    ):
        comparisons[name] = {
            "gta_backups_per_second_delta": _summary(
                condition_arrays[current]["gta_rate"]
                - condition_arrays[reference]["gta_rate"]
            ),
            "relative_throughput_delta": _summary(
                condition_arrays[current]["relative_throughput"]
                - condition_arrays[reference]["relative_throughput"]
            ),
            "gta_l1_recovery_auc_delta": _summary(
                condition_arrays[current]["l1_auc"]
                - condition_arrays[reference]["l1_auc"]
            ),
            "gta_linf_recovery_auc_delta": _summary(
                condition_arrays[current]["linf_auc"]
                - condition_arrays[reference]["linf_auc"]
            ),
        }
    return {
        "experiment": "arnoldi_cadence_ablation",
        "environment": "GARNET-200",
        "host": "DP",
        "reward_support": 20,
        "period_seconds": 5.0,
        "seeds": expected_seeds,
        "interpretation_scope": (
            "Reward shifts preserve the transition kernel. Initialization-only "
            "SLEM is exact for this protocol and does not test transition drift."
        ),
        "conditions": conditions,
        "paired_comparisons": comparisons,
    }


def plot(payload: dict[str, Any], output_stem: Path) -> None:
    apply_style()
    colors = get_palette(
        "line",
        ["relative_throughput", "l1_auc", "linf_auc"],
    )
    grid_color = get_utility_color("grid")
    conditions = payload["conditions"]
    x = np.arange(len(conditions))

    throughput = np.asarray(
        [row["relative_throughput"]["mean"] for row in conditions]
    )
    throughput_ci = np.asarray(
        [row["relative_throughput"]["ci95"] for row in conditions]
    )
    l1_auc = np.asarray([row["gta_l1_recovery_auc"]["mean"] for row in conditions])
    l1_ci = np.asarray([row["gta_l1_recovery_auc"]["ci95"] for row in conditions])
    linf_auc = np.asarray(
        [row["gta_linf_recovery_auc"]["mean"] for row in conditions]
    )
    linf_ci = np.asarray(
        [row["gta_linf_recovery_auc"]["ci95"] for row in conditions]
    )

    figure, left = plt.subplots(figsize=(8.2, 5.0))
    right = left.twinx()
    right.spines["right"].set_visible(True)

    left.errorbar(
        x,
        throughput,
        yerr=throughput_ci,
        color=colors["relative_throughput"],
        marker="o",
        capsize=4,
        label="GTA / Standard throughput",
    )
    right.errorbar(
        x,
        l1_auc,
        yerr=l1_ci,
        color=colors["l1_auc"],
        marker="s",
        capsize=4,
        label=r"$L_1$ recovery AUC",
    )
    right.errorbar(
        x,
        linf_auc,
        yerr=linf_ci,
        color=colors["linf_auc"],
        marker="^",
        capsize=4,
        label=r"$L_\infty$ recovery AUC",
    )

    left.set_xticks(x, [row["label"] for row in conditions])
    left.set_xlabel("Backups between Arnoldi refreshes")
    left.set_ylabel("Relative backup throughput (higher is better)")
    right.set_ylabel("GTA recovery AUC (lower is better)")
    left.set_ylim(0.0, 0.65)
    left.grid(True, color=grid_color)
    left.set_title("Arnoldi Cadence Ablation")

    refresh_text = [
        f"{row['slem_refreshes_per_period']['mean']:.1f} refreshes"
        for row in conditions
    ]
    for index, text in enumerate(refresh_text):
        left.annotate(
            text,
            (x[index], throughput[index]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=colors["relative_throughput"],
        )

    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="center right",
    )
    figure.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results" / "additional_evidence" / "arnoldi_cadence_30seed",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "additional_evidence" / "arnoldi_cadence_30seed",
    )
    args = parser.parse_args()

    payload = aggregate(args.input_dir)
    output_stem = args.out_dir / "arnoldi_cadence_ablation"
    output_stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    plot(payload, output_stem)
    print(f"results={output_stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()

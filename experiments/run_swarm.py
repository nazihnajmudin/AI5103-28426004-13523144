"""Percobaan BPSO pada instance kelompok (data/processed/instances).

Gap dihitung terhadap optimum brute-force, bukan optimal_cost dari paper.

Jalankan dari root repo:
    python experiments/run_swarm.py
    python experiments/run_swarm.py --seeds 10 --budget 3000
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.algorithm import ParticleSwarm, RunResult  # noqa: E402
from src.algorithm.baseline import RandomSearch, greedy_knapsack  # noqa: E402
from src.optimum import with_true_optimum  # noqa: E402
from src.problem import ProblemInstance, load_instance  # noqa: E402

DATA_DIR = ROOT / "data" / "processed" / "instances"
OUT_DIR = ROOT / "results" / "swarm"

# override dari default ParticleSwarm (sigmoid, w=0.7, c1=c2=1.5, v_max=4, S=40)
CONFIGS = {
    "sigmoid": {},
    "sigmoid_lin": {"inertia": 0.9, "inertia_end": 0.4},
    "vshape": {"transfer": "vshape"},
    "vshape_lin": {"transfer": "vshape", "inertia": 0.9, "inertia_end": 0.4},
    "sigmoid_s20": {"swarm_size": 20},
    "sigmoid_s80": {"swarm_size": 80},
    "sigmoid_v6": {"v_max": 6.0},
}

# konfigurasi yang ditampilkan di plot konvergensi
PLOT_CONFIGS = ["sigmoid", "sigmoid_lin", "vshape", "vshape_lin", "random"]
COLORS = {
    "sigmoid": "#2a78d6",
    "sigmoid_lin": "#eb6834",
    "vshape": "#1baf7a",
    "vshape_lin": "#eda100",
    "random": "#8a8984",
}
LABELS = {
    "sigmoid": "BPSO sigmoid",
    "sigmoid_lin": "BPSO sigmoid, inertia linear",
    "vshape": "BPSO V-shaped",
    "vshape_lin": "BPSO V-shaped, inertia linear",
    "random": "random search",
}

NEAR_OPT = 0.1  # gap (%) yang dianggap near-optimal


def batch_size(result: RunResult) -> int:
    return result.params.get("swarm_size", result.params.get("batch_size"))


def load_problems() -> tuple[list[ProblemInstance], dict[int, float]]:
    """Instance dengan optimum brute-force + gap solusi paper terhadap optimum tersebut"""
    problems, paper_gap = [], {}
    for path in DATA_DIR.glob("instance_*_diesets.json"):
        paper = load_instance(path)
        problem = with_true_optimum(paper)
        paper_gap[problem.n_diesets] = gap_pct(paper.optimal_cost, problem.optimal_cost)
        problems.append(problem)
    return sorted(problems, key=lambda p: p.n_diesets), paper_gap


def run_all(problems: list[ProblemInstance], seeds: int, budget: int) -> list[dict]:
    algorithms = {name: ParticleSwarm(max_iterations=10**6, max_evaluations=budget, **cfg)
                  for name, cfg in CONFIGS.items()}
    algorithms["random"] = RandomSearch(max_iterations=10**6, max_evaluations=budget)

    records = []
    for problem in problems:
        for name, algo in algorithms.items():
            for seed in range(seeds):
                result = algo.run(problem, seed=seed)
                result.save(OUT_DIR / "runs" / name / problem.key / f"seed_{seed:02d}.json")
                records.append(make_record(name, problem, result))
            print(f"[OK] {problem.key:<11} {name}")

        genome = greedy_knapsack(problem)
        obj = problem.evaluate(genome)
        records.append({
            "config": "greedy",
            "n_diesets": problem.n_diesets,
            "seed": 0,
            "best_cost": obj.cost,
            "gap_pct": gap_pct(obj.cost, problem.optimal_cost),
            "feasible": obj.feasible,
            "batch": 1,
            "evals_to_best": 1,
            "runtime_sec": 0.0,
            "history": [],
        })
    return records


def gap_pct(cost: float, optimal: float) -> float:
    return (cost - optimal) / optimal * 100


def make_record(name: str, problem, result: RunResult) -> dict:
    return {
        "config": name,
        "n_diesets": problem.n_diesets,
        "seed": result.seed,
        "best_cost": result.best_cost,
        "gap_pct": gap_pct(result.best_cost, problem.optimal_cost),
        "feasible": result.feasible,
        # iterasi 0 = inisialisasi, tiap iterasi memakai batch_size evaluasi
        "batch": batch_size(result),
        "evals_to_best": (result.best_iteration + 1) * batch_size(result),
        "runtime_sec": result.runtime_sec,
        "history": [gap_pct(c, problem.optimal_cost) for c in result.history_best_cost],
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(hit=df.gap_pct.abs() < 1e-6, near=df.gap_pct <= NEAR_OPT)
    summary = df.groupby(["n_diesets", "config"], sort=False).agg(
        runs=("seed", "size"),
        success_rate=("hit", "mean"),
        near_opt_rate=("near", "mean"),
        feasible_rate=("feasible", "mean"),
        gap_mean=("gap_pct", "mean"),
        gap_median=("gap_pct", "median"),
        gap_max=("gap_pct", "max"),
        evals_to_best=("evals_to_best", "mean"),
        runtime_sec=("runtime_sec", "mean"),
    )
    for col in ["success_rate", "near_opt_rate", "feasible_rate"]:
        summary[col] *= 100
    return summary.reset_index()


def plot_convergence(df: pd.DataFrame, paper_gap: dict[int, float]) -> Path:
    # n = 3 dan 9 sudah 100% optimum di semua konfigurasi, tidak informatif untuk diplot
    sizes = [n for n in sorted(df.n_diesets.unique()) if n >= 13]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, sizes):
        for name in PLOT_CONFIGS:
            runs = df[(df.n_diesets == n) & (df.config == name)]
            hist = np.maximum(np.array(runs.history.tolist()), 0.0)  # buang noise floating point
            x = np.arange(1, hist.shape[1] + 1) * runs.batch.iloc[0]
            ax.plot(x, np.median(hist, axis=0), color=COLORS[name], lw=2, label=LABELS[name])

        if paper_gap[n] > 1e-6:
            ax.axhline(paper_gap[n], color="#52514e", lw=1, ls="--")
            ax.text(x[0], paper_gap[n] * 0.8, "solusi paper", ha="left", va="top",
                    fontsize=9, color="#52514e")

        ax.set_title(f"{n} dieset", fontsize=11, loc="left")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_yticks([0, 1e-3, 1e-2, 1e-1, 1])
        ax.set_yticklabels(["0", "0,001", "0,01", "0,1", "1"])
        ax.set_ylim(0, 2)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("jumlah evaluasi")
    for ax in axes[:, 0]:
        ax.set_ylabel("median gap ke optimum (%)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Konvergensi BPSO vs random search, median 30 seed", fontsize=12, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    out = OUT_DIR / "convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budget", type=int, default=5000, help="max_evaluations per run")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems, paper_gap = load_problems()
    df = pd.DataFrame(run_all(problems, args.seeds, args.budget))

    summary = summarize(df)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    df.drop(columns=["history", "batch"]).to_csv(OUT_DIR / "runs.csv", index=False)
    plot_path = plot_convergence(df, paper_gap)

    pd.set_option("display.width", 200)
    print()
    print(summary.round(4).to_string(index=False))
    print(f"\nHasil di {OUT_DIR.relative_to(ROOT)}/ (summary.csv, runs.csv, {plot_path.name})")


if __name__ == "__main__":
    main()

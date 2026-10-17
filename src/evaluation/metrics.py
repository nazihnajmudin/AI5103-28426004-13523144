"""
Metrik mengikuti paper (Schmidbauer et al., 2025):
  - % feasible            (Fig. 3 paper)
  - % near-optimal        gap <= 1% dibagi jumlah run feasible (Fig. 4 paper)
  - rasio cost terbaik    cost feasible terendah / optimum (Fig. 5 paper)
  - optimality gap
Tambahan: waktu eksekusi dan jumlah iterasi/evaluasi
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithm.base import RunResult
from src.problem import OPTIMUM_OPTIONS, ProblemInstance

NEAR_OPTIMAL_PCT = 1.0  # gap <= 1%, definisi Schmidbauer et al. (2025)


def reference_cost(inst: ProblemInstance, optimum: str) -> float:
    """Optimum pembanding: "scip" (paper) atau brute-force ("brute-force" dan "both")."""
    if optimum not in OPTIMUM_OPTIONS:
        raise ValueError(f"optimum harus salah satu dari {OPTIMUM_OPTIONS}")
    return inst.optimal_cost if optimum == "scip" else inst.brute_force_optimal_cost


def gap_pct(cost: float, inst: ProblemInstance, optimum: str) -> float:
    ref = reference_cost(inst, optimum)
    return (cost - ref) / ref * 100


def runs_to_frame(results: list[RunResult], instances: list[ProblemInstance], optimum: str) -> pd.DataFrame:
    """Satu baris per run + metriknya"""
    by_key = {inst.key: inst for inst in instances}

    rows = []
    for r in results:
        inst = by_key[r.instance]
        ref = reference_cost(inst, optimum)
        gap = gap_pct(r.best_cost, inst, optimum) if r.feasible else np.nan
        rows.append(
            {
                "algorithm": r.algorithm,
                "config": r.config,
                "instance": r.instance,
                "n_diesets": r.n_diesets,
                "seed": r.seed,
                "feasible": bool(r.feasible),
                "cost": r.best_cost,
                "reference_cost": ref,
                "gap_pct": gap,
                "cost_ratio": r.best_cost / ref if r.feasible else np.nan,
                "near_optimal": bool(r.feasible and gap <= NEAR_OPTIMAL_PCT),
                "optimum_found": bool(r.feasible and gap <= 1e-9),
                "scip_gap_pct": gap_pct(inst.optimal_cost, inst, optimum),
                "runtime_sec": r.runtime_sec,
                "iterations": r.iterations,
                "iterations_to_best": r.best_iteration,
                "evaluations": r.n_evaluations,
                "evaluations_to_best": r.evaluations_to_best,
                "stop_reason": r.stop_reason,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan per (algoritma, konfigurasi, instance)"""
    rows = []
    for (algo, config, n), g in df.groupby(["algorithm", "config", "n_diesets"], sort=False):
        feas = g[g.feasible]
        n_feas = len(feas)
        rows.append(
            {
                "algorithm": algo,
                "config": config,
                "n_diesets": n,
                "n_runs": len(g),
                "pct_feasible": 100 * n_feas / len(g),
                "pct_near_optimal": 100 * feas.near_optimal.sum() / n_feas if n_feas else np.nan,
                "cost_ratio_best": feas.cost_ratio.min(),
                "cost_ratio_mean": feas.cost_ratio.mean(),
                "gap_mean": feas.gap_pct.mean(),
                "gap_median": feas.gap_pct.median(),
                "gap_std": feas.gap_pct.std(),
                "gap_max": feas.gap_pct.max(),
                "pct_optimum_found": 100 * g.optimum_found.mean(),
                "scip_gap_pct": g.scip_gap_pct.iloc[0],
                "scip_cost_ratio": 1 + g.scip_gap_pct.iloc[0] / 100,
                "runtime_mean": g.runtime_sec.mean(),
                "iterations_mean": g.iterations.mean(),
                "iterations_to_best_mean": g.iterations_to_best.mean(),
                "evaluations_mean": g.evaluations.mean(),
                "evaluations_to_best_mean": g.evaluations_to_best.mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["algorithm", "config", "n_diesets"], ignore_index=True)


def rank_configs(summary: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata metrik lintas instance per konfigurasi, diurutkan dari yang terbaik.
    Urutan: 
        1. % feasible tertinggi, 
        2. % near-optimal tertinggi,
        3. gap rata-rata terkecil.
    """
    cols = [
        "pct_feasible", "pct_near_optimal", "gap_mean", "gap_max", "pct_optimum_found",
        "runtime_mean", "iterations_to_best_mean", "evaluations_to_best_mean",
    ]
    ranking = summary.groupby(["algorithm", "config"], as_index=False)[cols].mean()
    ranking = ranking.sort_values(
        ["algorithm", "pct_feasible", "pct_near_optimal", "gap_mean"],
        ascending=[True, False, False, True],
        ignore_index=True,
    )
    ranking.insert(2, "rank", ranking.groupby("algorithm").cumcount() + 1)
    return ranking


def best_configs(ranking: pd.DataFrame) -> dict[str, str]:
    """Best config dari tabel `rank_configs`."""
    best = ranking[ranking["rank"] == 1]
    return dict(zip(best.algorithm, best.config))


def save_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

"""Statistik deskriptif gap dan uji signifikansi antar algoritma."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ranksums

ALPHA = 0.05


def gap_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    """Gap terbaik, rata-rata, simpangan baku, dan terburuk per (algoritma, instance)"""
    feas = runs[runs.feasible]
    stats = feas.groupby(["n_diesets", "algorithm"]).gap_pct.agg(
        best="min", mean="mean", std="std", worst="max", n_runs="size"
    )
    return stats.reset_index()


def rank_sum_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon rank-sum dua sampel independen -> (statistik, p-value).

    Kalau kedua sampel identik dan konstan (misalnya semua run mencapai optimum),
    tidak ada perbedaan yang bisa diuji, jadi p = 1.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(np.concatenate([a, b])) == 0:
        return 0.0, 1.0
    result = ranksums(a, b)
    return float(result.statistic), float(result.pvalue)


def pairwise_tests(runs: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Uji Wilcoxon rank-sum gap antar setiap pasangan algoritma, per instance.

    `runs` berisi satu konfigurasi per algoritma (misalnya best config).
    Statistik negatif berarti gap algoritma A cenderung lebih kecil daripada B.
    """
    feas = runs[runs.feasible]
    rows = []
    for n, g in feas.groupby("n_diesets"):
        for algo_a, algo_b in combinations(sorted(g.algorithm.unique()), 2):
            gap_a = g[g.algorithm == algo_a].gap_pct.to_numpy()
            gap_b = g[g.algorithm == algo_b].gap_pct.to_numpy()
            stat, p = rank_sum_test(gap_a, gap_b)
            rows.append(
                {
                    "n_diesets": n,
                    "algorithm_a": algo_a,
                    "algorithm_b": algo_b,
                    "gap_mean_a": gap_a.mean(),
                    "gap_mean_b": gap_b.mean(),
                    "statistic": stat,
                    "p_value": p,
                    "significant": p < alpha,
                    "better": (algo_a if gap_a.mean() < gap_b.mean() else algo_b) if p < alpha else "-",
                }
            )
    return pd.DataFrame(rows)

"""Statistik cost dan gap, serta uji signifikansi antar algoritma."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ALPHA = 0.05


def cost_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    """Cost terbaik, rata-rata, simpangan baku, dan terburuk per (algoritma, instance)"""
    feas = runs[runs.feasible]
    stats = feas.groupby(["n_diesets", "algorithm"]).cost.agg(
        best="min", mean="mean", std="std", worst="max", n_runs="size"
    )
    return stats.reset_index()


def gap_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    """Gap terbaik, rata-rata, simpangan baku, dan terburuk per (algoritma, instance)"""
    feas = runs[runs.feasible]
    stats = feas.groupby(["n_diesets", "algorithm"]).gap_pct.agg(
        best="min", mean="mean", std="std", worst="max", n_runs="size"
    )
    return stats.reset_index()


def pairwise_tests(runs: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Uji Mann-Whitney U (Wilcoxon rank-sum) gap antar pasangan algoritma, per instance.

    `runs` berisi satu konfigurasi per algoritma (best config). Jika p < alpha,
    algoritma dengan median gap lebih kecil dianggap lebih baik.
    """
    feas = runs[runs.feasible]
    algorithms = sorted(feas.algorithm.unique())
    rows = []
    for n, g in feas.groupby("n_diesets"):
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                a, b = algorithms[i], algorithms[j]
                gap_a = g[g.algorithm == a].gap_pct
                gap_b = g[g.algorithm == b].gap_pct
                p = mannwhitneyu(gap_a, gap_b).pvalue
                if np.isnan(p):
                    p = 1.0   # semua gap sama (misalnya semua run mencapai optimum)

                better = "-"
                if p < alpha:
                    better = a if gap_a.median() < gap_b.median() else b
                rows.append(
                    {
                        "n_diesets": n,
                        "algorithm_a": a,
                        "algorithm_b": b,
                        "median_a": gap_a.median(),
                        "median_b": gap_b.median(),
                        "p_value": p,
                        "significant": p < alpha,
                        "better": better,
                    }
                )
    return pd.DataFrame(rows)

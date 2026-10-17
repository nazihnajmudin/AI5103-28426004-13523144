"""Plot percobaan.

Dua jenis plot:
  1. per konfigurasi  : sebaran hasil antar seed, x = instance
  2. antar grup       : grup = konfigurasi (dalam satu algoritma) atau algoritma
                        (best config tiap algoritma), dipilih lewat argumen `by`

Gap dalam persen terhadap optimum pilihan `optimum`. Jika `optimum="both"`, solusi
SCIP (paper) ikut digambar sebagai garis putus-putus. Sumbu gap memakai skala
symlog karena gap-nya sangat kecil (sering < 0,01%) dan bisa 0.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithm.base import RunResult
from src.evaluation.metrics import reference_cost
from src.problem import ProblemInstance

LABELS = {
    "pct_feasible": "Run feasible (%)",
    "pct_near_optimal": "Near-optimal, gap <= 1% (%)",
    "cost_ratio_best": "Cost terbaik / optimum",
    "gap_mean": "Gap rata-rata (%)",
    "pct_optimum_found": "Optimum ditemukan (%)",
    "runtime_mean": "Waktu eksekusi (detik)",
    "iterations_to_best_mean": "Iterasi sampai solusi terbaik",
    "evaluations_to_best_mean": "Evaluasi sampai solusi terbaik",
    "gap_pct": "Gap ke optimum (%)",
    "runtime_sec": "Waktu eksekusi (detik)",
    "iterations_to_best": "Iterasi sampai solusi terbaik",
    "evaluations_to_best": "Evaluasi sampai solusi terbaik",
}
SCALABILITY_METRICS = [
    "pct_feasible", "pct_near_optimal", "cost_ratio_best", "gap_mean",
    "pct_optimum_found", "runtime_mean", "iterations_to_best_mean", "evaluations_to_best_mean",
]
GAP_LINTHRESH = 1e-3  # di bawah 0,001% sumbu gap dibuat linear
SCIP_STYLE = dict(color="dimgray", ls="--", lw=1.2, label="solusi SCIP (paper)")


def save_figure(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _gap_axis(ax, optimum: str) -> None:
    ax.set_yscale("symlog", linthresh=GAP_LINTHRESH)
    name = "SCIP" if optimum == "scip" else "brute-force"
    ax.set_ylabel(f"Gap ke optimum {name} (%)")


def _gap_history(result: RunResult, reference: float) -> np.ndarray:
    return (np.array(result.history_best_cost) - reference) / reference * 100


def _scip_gap(inst: ProblemInstance, optimum: str) -> float:
    ref = reference_cost(inst, optimum)
    return (inst.optimal_cost - ref) / ref * 100


def _figure_legend(fig, axes) -> None:
    """Satu legend di kanan figure, gabungan dari semua panel."""
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(h)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))


# ------------------------------------------------------------------ per konfigurasi
def plot_config_distribution(df: pd.DataFrame, title: str, optimum: str):
    """Satu konfigurasi: sebaran hasil 30 seed di tiap instance."""
    sizes = sorted(df.n_diesets.unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # % feasible dan % near-optimal per instance
    ax = axes[0, 0]
    x = np.arange(len(sizes))
    feas = [100 * df[df.n_diesets == n].feasible.mean() for n in sizes]
    near = []
    for n in sizes:
        f = df[(df.n_diesets == n) & df.feasible]
        near.append(100 * f.near_optimal.mean() if len(f) else np.nan)
    ax.bar(x - 0.2, feas, width=0.4, label="feasible")
    ax.bar(x + 0.2, near, width=0.4, label="near-optimal (dari yang feasible)")
    ax.set_xticks(x, sizes)
    ax.set_ylim(0, 130)
    ax.set_ylabel("Persentase run (%)")
    ax.set_title("Feasible dan near-optimal")
    ax.legend(loc="upper center", ncol=2, fontsize=8)

    panels = [
        (axes[0, 1], "gap_pct", "Gap ke optimum"),
        (axes[1, 0], "runtime_sec", "Waktu eksekusi"),
        (axes[1, 1], "evaluations_to_best", "Evaluasi sampai solusi terbaik"),
    ]
    for ax, metric, panel_title in panels:
        data = [df[df.n_diesets == n][metric].dropna() for n in sizes]
        ax.boxplot(data, tick_labels=sizes)
        ax.set_ylabel(LABELS[metric])
        ax.set_title(panel_title)
        if metric == "gap_pct":
            _gap_axis(ax, optimum)
            if optimum == "both":
                # garis pendek di tiap kotak: posisi solusi SCIP untuk instance itu
                scip = [df[df.n_diesets == n].scip_gap_pct.iloc[0] for n in sizes]
                ax.hlines(scip, np.arange(1, len(sizes) + 1) - 0.4, np.arange(1, len(sizes) + 1) + 0.4, **SCIP_STYLE)
                ax.legend(fontsize=8)
        if metric == "runtime_sec":
            ax.set_yscale("log")

    for ax in axes.ravel():
        ax.set_xlabel("Jumlah dieset flexible")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_seed_convergence(results: list[RunResult], instances: list[ProblemInstance], title: str, optimum: str):
    """Satu konfigurasi: kurva konvergensi tiap seed (tipis) dan mediannya (tebal)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, inst in zip(axes.ravel(), instances):
        ref = reference_cost(inst, optimum)
        runs = [r for r in results if r.instance == inst.key]
        for r in runs:
            ax.plot(r.history_evaluations, _gap_history(r, ref), color="C0", lw=0.6, alpha=0.3)
        x, median = _median_curve(runs, ref)
        ax.plot(x, median, color="C3", lw=2, label="median")
        if optimum == "both":
            ax.axhline(_scip_gap(inst, optimum), **SCIP_STYLE)
        ax.set_title(f"{inst.n_diesets} dieset")
        ax.set_xlabel("Jumlah evaluasi")
        _gap_axis(ax, optimum)
    fig.suptitle(title)
    fig.tight_layout()
    _figure_legend(fig, axes)
    return fig


# ------------------------------------------------------------------ antar grup
def _median_curve(runs: list[RunResult], reference: float) -> tuple[np.ndarray, np.ndarray]:
    """Median gap per iterasi. Run yang lebih pendek diperpanjang dengan nilai terakhirnya."""
    longest = max(runs, key=lambda r: len(r.history_evaluations))
    length = len(longest.history_evaluations)
    curves = []
    for r in runs:
        gap = _gap_history(r, reference)
        curves.append(np.concatenate([gap, np.full(length - len(gap), gap[-1])]))
    return np.array(longest.history_evaluations), np.median(curves, axis=0)


def plot_group_convergence(results: list[RunResult], instances: list[ProblemInstance], by: str, title: str, optimum: str):
    """Median konvergensi tiap grup (`by` = "config" atau "algorithm"), satu panel per instance."""
    groups = sorted({getattr(r, by) for r in results})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, inst in zip(axes.ravel(), instances):
        ref = reference_cost(inst, optimum)
        for k, group in enumerate(groups):
            runs = [r for r in results if r.instance == inst.key and getattr(r, by) == group]
            x, median = _median_curve(runs, ref)
            ax.plot(x, median, color=f"C{k}", lw=1.8, label=group)
        if optimum == "both":
            ax.axhline(_scip_gap(inst, optimum), **SCIP_STYLE)
        ax.set_title(f"{inst.n_diesets} dieset")
        ax.set_xlabel("Jumlah evaluasi")
        _gap_axis(ax, optimum)
    fig.suptitle(title)
    fig.tight_layout()
    _figure_legend(fig, axes)
    return fig


def plot_group_scalability(summary: pd.DataFrame, by: str, title: str, optimum: str,
                           metrics: list[str] = SCALABILITY_METRICS):
    """Metrik ringkasan terhadap jumlah dieset, satu garis per grup."""
    groups = sorted(summary[by].unique())
    sizes = sorted(summary.n_diesets.unique())
    scip = summary.groupby("n_diesets")[["scip_gap_pct", "scip_cost_ratio"]].first().loc[sizes]

    nrows = int(np.ceil(len(metrics) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.2 * nrows))
    for ax, metric in zip(axes.ravel(), metrics):
        for k, group in enumerate(groups):
            g = summary[summary[by] == group].sort_values("n_diesets")
            ax.plot(g.n_diesets, g[metric], marker="o", color=f"C{k}", label=group)
        ax.set_title(LABELS[metric])
        ax.set_xlabel("Jumlah dieset flexible")
        ax.set_xticks(sizes)
        if metric.startswith("pct_"):
            ax.set_ylim(-5, 105)
        if metric == "gap_mean":
            if optimum == "both":
                ax.plot(sizes, scip.scip_gap_pct, marker="x", **SCIP_STYLE)
            _gap_axis(ax, optimum)
        if metric == "cost_ratio_best":
            if optimum == "both":
                ax.plot(sizes, scip.scip_cost_ratio, marker="x", **SCIP_STYLE)
            ax.ticklabel_format(axis="y", useOffset=False)
        if metric == "runtime_mean":
            ax.set_yscale("log")
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    _figure_legend(fig, axes)
    return fig


def plot_group_distribution(df: pd.DataFrame, metric: str, by: str, title: str, optimum: str):
    """Sebaran satu metrik antar seed, x = grup, satu panel per instance."""
    groups = sorted(df[by].unique())
    sizes = sorted(df.n_diesets.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, n in zip(axes.ravel(), sizes):
        data = [df[(df.n_diesets == n) & (df[by] == g)][metric].dropna() for g in groups]
        ax.boxplot(data, tick_labels=groups)
        ax.set_title(f"{n} dieset")
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_ylabel(LABELS[metric])
        if metric == "gap_pct":
            if optimum == "both":
                ax.axhline(df[df.n_diesets == n].scip_gap_pct.iloc[0], **SCIP_STYLE)
            _gap_axis(ax, optimum)
        if metric == "runtime_sec":
            ax.set_yscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    _figure_legend(fig, axes)
    return fig

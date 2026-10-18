"""Demo langsung: GA, BPSO, dan random search pada satu instance.

Ketiga algoritma dijalankan bergantian, iterasi per iterasi, dengan seed dan budget yang
sama seperti notebook percobaan. Dipakai oleh `experiments/demo.ipynb`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.algorithm import GeneticAlgorithm, ParticleSwarm, RandomSearch
from src.algorithm.base import Algorithm
from src.evaluation.runner import make_seeds
from src.optimum import brute_force
from src.paths import INSTANCES_DIR
from src.problem import ProblemInstance, load_instance

MASTER_SEED = 2026
COLORS = {"GA": "#eb6834", "BPSO": "#2a78d6", "Random": "#8a8984"}
MACHINE_COLORS = ["#d9d4c7", "#4a4a48"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


@dataclass
class Lane:
    """Satu algoritma yang sedang berjalan"""

    label: str
    config: str
    algo: Algorithm
    steps: object = None
    done: bool = False
    elapsed: float = 0.0
    evals: list = field(default_factory=list)
    gaps: list = field(default_factory=list)


def stepper(algo: Algorithm, problem: ProblemInstance, seed: int):
    """Sama seperti Algorithm.run(), tetapi berhenti setiap iterasi supaya bisa dianimasikan"""
    algo._bind(problem, seed)
    algo.initialize()
    yield
    while algo._stop_reason() is None:
        algo.iteration += 1
        algo.iterate()
        yield


def fmt_num(x: float, digits: int = 2) -> str:
    """Format angka Indonesia: 6.112.960,59"""
    return f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def build_lanes(budget: int) -> list[Lane]:
    # konfigurasi terbaik tiap algoritma, budget sama dengan notebook percobaan
    limits = dict(max_evaluations=budget, max_iterations=1000, patience=None)
    return [
        Lane("GA", "sel_rank", GeneticAlgorithm(selection="rank", **limits)),
        Lane("BPSO", "sigmoid_s20", ParticleSwarm(swarm_size=20, **limits)),
        Lane("Random", "batch_20", RandomSearch(batch_size=20, **limits)),
    ]


class Demo:
    """Satu sesi demo: `frames()` menjalankan algoritma dan memperbarui `fig` setiap frame"""

    def __init__(self, n_diesets: int = 19, seed: int | None = None, budget: int = 5000, step: int = 50):
        self.problem = load_instance(INSTANCES_DIR / f"instance_{n_diesets}_diesets.json", optimum="both")
        self.seed = seed if seed is not None else make_seeds(MASTER_SEED, 1)[0]
        self.budget = budget
        self.step = step

        self.opt_cost, self.opt_genome, _ = brute_force(self.problem)
        self.scip_gap = self.gap(self.problem.optimal_cost)

        self.lanes = build_lanes(budget)
        for lane in self.lanes:
            lane.steps = stepper(lane.algo, self.problem, self.seed)
        self._setup_figure()

    def gap(self, cost: float) -> float:
        return (cost - self.opt_cost) / self.opt_cost * 100

    def frames(self):
        """Jalankan ketiga algoritma bergantian, sampai jumlah evaluasi yang sama tiap frame"""
        target = 0
        while not all(lane.done for lane in self.lanes):
            target += self.step
            for lane in self.lanes:
                self._advance(lane, target)
            self._update_figure()
            yield target

    def summary(self) -> pd.DataFrame:
        rows = []
        for lane in self.lanes:
            obj = self.problem.evaluate(lane.algo.best_genome)
            rows.append({
                "algoritma": lane.label,
                "konfigurasi": lane.config,
                "cost": obj.cost,
                "gap_pct": self.gap(obj.cost),
                "feasible": obj.feasible,
                "evaluasi": lane.algo.n_evaluations,
                "evaluasi_sampai_terbaik": lane.algo.best_evaluation,
                "waktu_sec": lane.elapsed,
            })
        return pd.DataFrame(rows)

    def _advance(self, lane: Lane, target: int) -> None:
        while not lane.done and (not lane.evals or lane.algo.n_evaluations < target):
            started = perf_counter()
            try:
                next(lane.steps)
            except StopIteration:
                lane.done = True
            lane.elapsed += perf_counter() - started
            lane.evals.append(lane.algo.n_evaluations)
            lane.gaps.append(self.gap(lane.algo.best_cost))

    def _machine_of(self, genome: np.ndarray) -> np.ndarray:
        return self.problem.machine_index[np.arange(self.problem.n_diesets), genome]

    def _setup_figure(self) -> None:
        problem, lanes = self.problem, self.lanes
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.55], width_ratios=[1.25, 1], hspace=0.55, wspace=0.18)
        ax_conv = fig.add_subplot(gs[0:2, 0])
        ax_grid = fig.add_subplot(gs[0, 1])
        ax_load = fig.add_subplot(gs[1, 1])
        ax_table = fig.add_subplot(gs[2, :])
        fig.suptitle(f"Demo: alokasi {problem.n_diesets} dieset ke {problem.n_machines} mesin, "
                     f"budget {fmt_num(self.budget, 0)} evaluasi, seed {self.seed}", fontsize=13, color=INK)

        # konvergensi
        self.lines = {}
        for lane in lanes:
            (self.lines[lane.label],) = ax_conv.plot([], [], color=COLORS[lane.label], lw=2,
                                                     label=f"{lane.label} ({lane.config})")
        if self.scip_gap > 1e-9:
            ax_conv.axhline(self.scip_gap, color=MUTED, lw=1, ls="--")
            ax_conv.text(self.budget * 0.02, self.scip_gap * 0.8, "solusi SCIP (paper)",
                         va="top", fontsize=9, color=MUTED)
        ax_conv.set_xlim(0, self.budget)
        ax_conv.set_yscale("symlog", linthresh=1e-3)
        ax_conv.set_ylim(0, 3)
        ax_conv.set_yticks([0, 1e-3, 1e-2, 1e-1, 1])
        ax_conv.set_yticklabels(["0", "0,001", "0,01", "0,1", "1"])
        ax_conv.set_xlabel("jumlah evaluasi")
        ax_conv.set_ylabel("gap ke optimum (%)")
        ax_conv.set_title("Konvergensi", loc="left", fontsize=11)
        ax_conv.grid(True, color=GRID, lw=0.8)
        ax_conv.set_axisbelow(True)
        ax_conv.legend(loc="upper right", frameon=False, fontsize=9)
        for side in ["top", "right"]:
            ax_conv.spines[side].set_visible(False)

        # alokasi dieset -> mesin
        rows = ["Optimum"] + [lane.label for lane in lanes]
        matrix = np.tile(self._machine_of(self.opt_genome), (len(rows), 1))
        self.image = ax_grid.imshow(matrix, cmap=ListedColormap(MACHINE_COLORS), vmin=0, vmax=1, aspect="auto")
        ax_grid.set_yticks(range(len(rows)), rows)
        ax_grid.set_xticks(range(problem.n_diesets), [str(i + 1) for i in range(problem.n_diesets)], fontsize=8)
        ax_grid.set_xlabel("dieset")
        ax_grid.set_title("Alokasi solusi terbaik (× = beda dari optimum)", loc="left", fontsize=11)
        ax_grid.tick_params(length=0)
        ax_grid.legend(handles=[Patch(color=c, label=n) for c, n in zip(MACHINE_COLORS, problem.machine_names)],
                       loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)

        # beban mesin
        x = np.arange(problem.n_machines)
        width = 0.8 / len(lanes)
        self.bars = {}
        for k, lane in enumerate(lanes):
            self.bars[lane.label] = ax_load.bar(x + (k - 1) * width, np.zeros(problem.n_machines), width * 0.9,
                                                color=COLORS[lane.label])
        ax_load.axhline(100, color=MUTED, lw=1, ls="--")
        ax_load.set_xticks(x, [f"{n}\nkapasitas {fmt_num(c, 0)} jam"
                               for n, c in zip(problem.machine_names, problem.capacity)])
        ax_load.set_ylim(0, 115)
        ax_load.set_ylabel("beban (% kapasitas)")
        ax_load.set_title("Beban mesin solusi terbaik", loc="left", fontsize=11)
        for side in ["top", "right"]:
            ax_load.spines[side].set_visible(False)

        # tabel hasil
        ax_table.axis("off")
        header = ["Algoritma", "Konfigurasi", "Cost", "Gap (%)", "Feasible", "Evaluasi", "Waktu (s)"]
        self.table = ax_table.table(cellText=[[lane.label, lane.config, "", "", "", "", ""] for lane in lanes],
                                    colLabels=header, loc="center", cellLoc="center")
        self.table.auto_set_font_size(False)
        self.table.set_fontsize(10)
        self.table.scale(1, 1.5)
        for (row, _), cell in self.table.get_celld().items():
            cell.set_edgecolor(GRID)
            if row == 0:
                cell.set_text_props(weight="bold")

        self.fig, self.ax_grid, self.ax_load = fig, ax_grid, ax_load
        self.marks: list = []

    def _update_figure(self) -> None:
        problem = self.problem
        opt_machine = self._machine_of(self.opt_genome)
        matrix = [opt_machine]

        for mark in self.marks:
            mark.remove()
        self.marks = []

        for k, lane in enumerate(self.lanes):
            algo = lane.algo
            self.lines[lane.label].set_data(lane.evals, np.maximum(lane.gaps, 0.0))

            machine = self._machine_of(algo.best_genome)
            matrix.append(machine)
            for i in np.flatnonzero(machine != opt_machine):
                self.marks.append(self.ax_grid.text(i, k + 1, "×", ha="center", va="center",
                                                    color="#e34948", fontsize=12, weight="bold"))

            obj = problem.evaluate(algo.best_genome)
            for rect, load, cap in zip(self.bars[lane.label], obj.loads, problem.capacity):
                rect.set_height(100 * load / cap)
                self.marks.append(self.ax_load.text(rect.get_x() + rect.get_width() / 2, 100 * load / cap + 1.5,
                                                    fmt_num(load, 0), ha="center", va="bottom",
                                                    fontsize=8, color=MUTED))

            values = [fmt_num(obj.cost), fmt_num(lane.gaps[-1], 4), "ya" if obj.feasible else "tidak",
                      fmt_num(algo.n_evaluations, 0), fmt_num(lane.elapsed, 3)]
            for col, value in enumerate(values, start=2):
                self.table[k + 1, col].get_text().set_text(value)

        self.image.set_data(np.array(matrix))

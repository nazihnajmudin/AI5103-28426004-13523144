from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Objective:
    """Hasil evaluasi satu genome"""

    cost: float
    loads: np.ndarray           # beban kerja per mesin, jam
    overload: np.ndarray        # pelanggaran kapasitas per mesin, jam
    feasible: bool

    @property
    def total_overload(self) -> float:
        return float(self.overload.sum())


@dataclass(frozen=True)
class ProblemInstance:
    """Satu instance dataset dalam bentuk matriks.

    Matriks (n_diesets, n_choices):
      - `cost[i, j]`           biaya alokasi dieset i ke pilihan j
      - `work[i, j]`           beban kerja dieset i pada pilihan j
      - `machine_index[i, j]`  indeks mesin (ke `capacity`) dari pilihan j
    """

    key: str
    n_diesets: int
    n_machines: int
    machine_ids: np.ndarray
    machine_names: list[str]
    capacity: np.ndarray          # (n_machines,)
    dieset_ids: list[int]
    choice_counts: np.ndarray     # (n_diesets,) jumlah mesin feasible per dieset
    cost: np.ndarray              # (n_diesets, n_choices)
    work: np.ndarray              # (n_diesets, n_choices)
    machine_index: np.ndarray     # (n_diesets, n_choices)
    optimal_cost: float           # reference optimal dari paper
    optimal_genome: np.ndarray | None = field(default=None, repr=False)
    feasibility_tolerance: float = 1e-6

    @property
    def n_choices(self) -> int:
        return int(self.cost.shape[1])

    @property
    def search_space_size(self) -> float:
        return float(np.prod(self.choice_counts.astype(float)))

    def clip(self, genomes: np.ndarray) -> np.ndarray:
        """Menjaga tiap gen pada rentang valid"""
        g = np.asarray(genomes, dtype=np.int64)
        return np.clip(g, 0, self.choice_counts - 1)

    def evaluate_batch(self, genomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluasi kumpulan genome (satu baris = satu genome)
        Returns: (cost, loads), cost (P,), loads (P, n_machines)
        """
        g = self.clip(np.atleast_2d(genomes))
        rows = np.arange(self.n_diesets)
        cost = self.cost[rows, g].sum(axis=1)
        work_sel = self.work[rows, g]
        machine_sel = self.machine_index[rows, g]

        # Jumlah beban kerja per mesin
        loads = np.zeros((g.shape[0], self.n_machines))
        for m in range(self.n_machines):
            loads[:, m] = np.where(machine_sel == m, work_sel, 0.0).sum(axis=1)

        return cost, loads

    def evaluate(self, genome: np.ndarray) -> Objective:
        """Evaluasi satu genome menjadi Objective"""
        cost, loads = self.evaluate_batch(genome)
        loads = loads[0]
        overload = np.maximum(0.0, loads - self.capacity)
        return Objective(
            cost=float(cost[0]),
            loads=loads,
            overload=overload,
            feasible=bool(np.all(overload <= self.feasibility_tolerance)),
        )

    def assignment(self, genome: np.ndarray) -> dict[int, int]:
        """Genome -> {dieset_id: machine_id}"""
        g = self.clip(np.asarray(genome)).ravel()
        return {
            int(self.dieset_ids[i]): int(self.machine_ids[self.machine_index[i, g[i]]])
            for i in range(self.n_diesets)
        }

    def cost_bounds(self) -> tuple[float, float]:
        """Batas bawah/atas cost (tanpa capacity constraint)"""
        lo, hi = 0.0, 0.0
        for i in range(self.n_diesets):
            valid = self.cost[i, : self.choice_counts[i]]
            lo += valid.min()
            hi += valid.max()
        return float(lo), float(hi)


def load_instance(path: str | Path) -> ProblemInstance:
    """Baca satu file: `data/processed/instances/instance_*.json`"""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    machines = raw["machines"]
    machine_ids = np.array([m["machine_id"] for m in machines], dtype=np.int64)
    machine_pos = {int(mid): i for i, mid in enumerate(machine_ids)}
    capacity = np.array([m["capacity_hours"] for m in machines], dtype=float)

    diesets = raw["diesets"]
    n_diesets = len(diesets)
    n_choices = max(len(d["choices"]) for d in diesets)

    cost = np.full((n_diesets, n_choices), np.inf)
    work = np.full((n_diesets, n_choices), np.inf)
    machine_index = np.zeros((n_diesets, n_choices), dtype=np.int64)
    choice_counts = np.zeros(n_diesets, dtype=np.int64)

    # Isi matriks cost/work/machine_index per pilihan mesin
    for i, d in enumerate(diesets):
        choice_counts[i] = len(d["choices"])
        for j, c in enumerate(d["choices"]):
            cost[i, j] = c["total_cost"]
            work[i, j] = c["work_hours"]
            machine_index[i, j] = machine_pos[int(c["machine_id"])]

    gt = raw["ground_truth"]
    optimal_genome = None
    gt_assignment = gt.get("optimal_assignment_dieset_to_machine")
    if gt_assignment:
        # Buat optimal genome dari assignment ground truth
        optimal_genome = np.zeros(n_diesets, dtype=np.int64)
        for i, d in enumerate(diesets):
            target = int(gt_assignment[str(d["dieset_id"])])
            optimal_genome[i] = next(
                j for j, c in enumerate(d["choices"]) if int(c["machine_id"]) == target
            )

    return ProblemInstance(
        key=f"{raw['n_diesets']}_diesets",
        n_diesets=n_diesets,
        n_machines=len(machines),
        machine_ids=machine_ids,
        machine_names=[m["machine_name"] for m in machines],
        capacity=capacity,
        dieset_ids=[int(d["dieset_id"]) for d in diesets],
        choice_counts=choice_counts,
        cost=cost,
        work=work,
        machine_index=machine_index,
        optimal_cost=float(gt["optimal_cost"]),
        optimal_genome=optimal_genome,
    )


def load_instances(directory: str | Path | None = None) -> list[ProblemInstance]:
    """Baca semua instance: kecil -> besar (3 -> 19 dieset)"""
    from src.paths import INSTANCES_DIR

    directory = Path(directory) if directory is not None else INSTANCES_DIR
    instances = [load_instance(p) for p in directory.glob("instance_*_diesets.json")]
    return sorted(instances, key=lambda inst: inst.n_diesets)

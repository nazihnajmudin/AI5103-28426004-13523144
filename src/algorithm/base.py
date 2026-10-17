from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from src.problem import ProblemInstance


@dataclass
class RunResult:
    """Hasil satu run algoritma pada satu instance dan satu seed"""

    algorithm: str
    params: dict
    instance: str
    n_diesets: int
    seed: int

    best_genome: list[int]
    best_cost: float            # objective cost, tanpa penalty
    best_fitness: float         # cost + penalty capacity
    feasible: bool
    overload_hours: float

    iterations: int
    n_evaluations: int
    runtime_sec: float
    stop_reason: str

    history_best_cost: list[float] = field(default_factory=list, repr=False)
    history_evaluations: list[int] = field(default_factory=list, repr=False)
    best_iteration: int = 0
    evaluations_to_best: int = 0
    config: str = "default"     # nama konfigurasi, diisi oleh runner
    machine_loads: list[float] = field(default_factory=list)
    assignment: dict[str, int] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RunResult":
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class Algorithm:
    """Must Implements: `name`, `initialize()`, `iterate()`

    Inherent for subclass:
        self.n_genes                  panjang genome
        self.choice_counts            banyak pilihan valid per gen
        self.rng                      np.random.Generator
        self.random_genomes(n)        n genome acak valid
        self.evaluate(genomes)        fitness, makin kecil makin baik
        self.iteration                nomor iterasi berjalan
    """

    # Wajib diisi subclass
    name: str = "algorithm"

    def __init__(
        self,
        *,
        max_iterations: int = 200,
        max_evaluations: int | None = None,
        patience: int | None = None,        # jumlah evaluasi tanpa perbaikan, None = mati
        penalty_weight: float | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_evaluations = max_evaluations
        self.patience = patience
        self.penalty_weight = penalty_weight

        self.problem: ProblemInstance | None = None
        self.rng: np.random.Generator | None = None

    # Wajib diisi subclass
    def initialize(self) -> None:
        raise NotImplementedError

    def iterate(self) -> None:
        raise NotImplementedError

    def extra_params(self) -> dict:
        """Hyperparameter khusus subclass"""
        return {}

    # helper untuk subclass
    @property
    def n_genes(self) -> int:
        return self.problem.n_diesets

    @property
    def choice_counts(self) -> np.ndarray:
        return self.problem.choice_counts

    def random_genomes(self, n: int) -> np.ndarray:
        """n genome acak valid"""
        return self.rng.integers(0, self.choice_counts, size=(n, self.n_genes))

    def evaluate(self, genomes: np.ndarray) -> np.ndarray:
        """Fitness tiap genome (P,) -> makin kecil makin baik"""
        g = self.problem.clip(np.atleast_2d(genomes))
        cost, loads = self.problem.evaluate_batch(g)
        overload = np.maximum(0.0, loads - self.problem.capacity)
        violated = np.any(overload > self.problem.feasibility_tolerance, axis=1)

        # Capacity penalty: besar overload + feasibility-first (ada/tidak)
        fitness = cost + self._penalty * ((overload**2).sum(axis=1) + violated)

        self.n_evaluations += len(g)
        i = int(np.argmin(fitness))
        if fitness[i] < self.best_fitness:
            self.best_fitness = float(fitness[i])
            self.best_cost = float(cost[i])
            self.best_genome = g[i].copy()
            self.best_iteration = self.iteration
            self.best_evaluation = self.n_evaluations

        return fitness

    def run(self, problem: ProblemInstance, seed: int) -> RunResult:
        """Jalankan algoritma satu kali pada `problem` dengan `seed` tertentu."""
        self._bind(problem, seed)

        started = perf_counter()
        self.initialize()
        self._history.append(self.best_cost)
        self._history_evals.append(self.n_evaluations)

        # Main Loop: iterate sampai salah satu kriteria stop terpenuhi
        stop_reason = self._stop_reason()
        while stop_reason is None:
            self.iteration += 1
            self.iterate()
            self._history.append(self.best_cost)
            self._history_evals.append(self.n_evaluations)
            stop_reason = self._stop_reason()
        runtime = perf_counter() - started

        objective = problem.evaluate(self.best_genome)
        return RunResult(
            algorithm=self.name,
            params=self.params(),
            instance=problem.key,
            n_diesets=problem.n_diesets,
            seed=int(seed),
            best_genome=[int(x) for x in self.best_genome],
            best_cost=objective.cost,
            best_fitness=self.best_fitness,
            feasible=objective.feasible,
            overload_hours=objective.total_overload,
            iterations=self.iteration,
            n_evaluations=self.n_evaluations,
            runtime_sec=runtime,
            stop_reason=stop_reason,
            history_best_cost=[float(x) for x in self._history],
            history_evaluations=list(self._history_evals),
            best_iteration=self.best_iteration,
            evaluations_to_best=self.best_evaluation,
            machine_loads=[float(x) for x in objective.loads],
            assignment={str(k): v for k, v in problem.assignment(self.best_genome).items()},
        )

    def params(self) -> dict:
        """Info hyperparameter (base + subclass)"""
        base = {
            "max_iterations": self.max_iterations,
            "max_evaluations": self.max_evaluations,
            "patience": self.patience,
            "penalty_weight": self.penalty_weight,
        }
        base.update(self.extra_params())
        return base

    def _bind(self, problem: ProblemInstance, seed: int) -> None:
        """Reset state -> satu objek algoritma bisa dipakai ulang"""
        self.problem = problem
        self.rng = np.random.default_rng(seed)
        self._penalty = (
            float(self.penalty_weight)
            if self.penalty_weight is not None
            else float(problem.cost_bounds()[1])
        )
        self.iteration = 0
        self.n_evaluations = 0
        self.best_genome = np.zeros(problem.n_diesets, dtype=np.int64)
        self.best_fitness = np.inf
        self.best_cost = np.inf
        self.best_iteration = 0
        self.best_evaluation = 0
        self._history: list[float] = []
        self._history_evals: list[int] = []

    def _stop_reason(self) -> str | None:
        if self.iteration >= self.max_iterations:
            return "max_iterations"
        if self.max_evaluations is not None and self.n_evaluations >= self.max_evaluations:
            return "max_evaluations"
        if self.patience is not None and self.n_evaluations - self.best_evaluation >= self.patience:
            return "patience"
        return None

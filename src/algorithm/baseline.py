from __future__ import annotations

import numpy as np

from src.algorithm.base import Algorithm
from src.problem import ProblemInstance


class RandomSearch(Algorithm):
    """Baseline: sampling genome acak, budget evaluasi sama dengan algoritma lain"""

    name = "random"

    def __init__(self, *, batch_size: int = 40, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def extra_params(self) -> dict:
        return {"batch_size": self.batch_size}

    def initialize(self) -> None:
        self.evaluate(self.random_genomes(self.batch_size))

    def iterate(self) -> None:
        self.evaluate(self.random_genomes(self.batch_size))


def greedy_knapsack(problem: ProblemInstance) -> np.ndarray:
    """Baseline greedy: pindahkan dieset ke choices[1] urut dari rasio penghematan/jam terbesar.

    Dieset hanya dipindah kalau mesin tujuan masih muat. Hasil bisa infeasible
    kalau mesin choices[0] sudah overload sejak awal, jadi cek dengan problem.evaluate().
    """
    saving = problem.cost[:, 0] - problem.cost[:, 1]
    ratio = saving / problem.work[:, 1]

    genome = np.zeros(problem.n_diesets, dtype=np.int64)
    for i in np.argsort(-ratio):
        if saving[i] <= 0:
            break
        trial = genome.copy()
        trial[i] = 1
        _, loads = problem.evaluate_batch(trial)
        m = problem.machine_index[i, 1]
        if loads[0, m] <= problem.capacity[m] + problem.feasibility_tolerance:
            genome = trial
    return genome

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.problem import ProblemInstance


def brute_force(problem: ProblemInstance, chunk: int = 1 << 16) -> tuple[float, np.ndarray, int]:
    """Enumerasi semua genome -> (optimal_cost, optimal_genome, n_feasible)

    Dipakai karena optimal_cost dari paper bukan optimum untuk instance 16, 18, 19.
    Ruang pencarian paling besar 2^19, jadi masih cepat.
    """
    total = int(problem.search_space_size)
    best_cost, best_genome, n_feasible = np.inf, None, 0

    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total))
        genomes = np.stack(np.unravel_index(idx, problem.choice_counts), axis=1)
        cost, loads = problem.evaluate_batch(genomes)
        feasible = np.all(loads <= problem.capacity + problem.feasibility_tolerance, axis=1)
        n_feasible += int(feasible.sum())

        if feasible.any():
            i = int(np.argmin(np.where(feasible, cost, np.inf)))
            if cost[i] < best_cost:
                best_cost, best_genome = float(cost[i]), genomes[i].copy()

    if best_genome is None:
        raise ValueError(f"{problem.key}: tidak ada solusi feasible")
    return best_cost, best_genome, n_feasible


def with_true_optimum(problem: ProblemInstance) -> ProblemInstance:
    """Salinan problem dengan optimal_cost/optimal_genome hasil brute-force"""
    cost, genome, _ = brute_force(problem)
    return replace(problem, optimal_cost=cost, optimal_genome=genome)

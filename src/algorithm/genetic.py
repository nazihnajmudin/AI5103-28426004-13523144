from __future__ import annotations

import numpy as np

from src.algorithm.base import Algorithm


class GeneticAlgorithm(Algorithm):
    name = "genetic"

    def __init__(
        self,
        *,
        population_size: int = 60,
        crossover_rate: float = 0.9,
        mutation_rate: float | None = None,   # None -> 1 / n_genes
        tournament_size: int = 3,
        elitism: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

    def extra_params(self) -> dict:
        return {
            "population_size": self.population_size,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "tournament_size": self.tournament_size,
            "elitism": self.elitism,
        }

    def initialize(self) -> None:
        # TODO: self.population = self.random_genomes(population_size); self.fitness = self.evaluate(...)
        raise NotImplementedError

    def iterate(self) -> None:
        # TODO: seleksi -> crossover -> mutasi -> evaluate -> populasi baru (pertahankan `elitism` terbaik)
        raise NotImplementedError

    def _select(self) -> np.ndarray:
        # TODO: tournament selection, kembalikan indeks parent terpilih
        raise NotImplementedError

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        # TODO: uniform/one-point crossover sesuai crossover_rate
        raise NotImplementedError

    def _mutate(self, offspring: np.ndarray) -> np.ndarray:
        # TODO: flip gen ke pilihan mesin lain dengan peluang mutation_rate (default 1/n_genes)
        raise NotImplementedError

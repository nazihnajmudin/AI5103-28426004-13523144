from __future__ import annotations

import numpy as np

from src.algorithm.base import Algorithm

SELECTIONS = ("tournament", "roulette", "rank")
CROSSOVERS = ("one_point", "two_point", "uniform")
MUTATIONS = ("bit_flip", "swap")


class GeneticAlgorithm(Algorithm):
    """GA generasional: seleksi -> crossover -> mutasi, dengan elitisme
    """

    name = "genetic"

    def __init__(
        self,
        *,
        population_size: int = 40,
        selection: str = "tournament",
        crossover: str = "uniform",
        mutation: str = "bit_flip",
        crossover_rate: float = 0.9,
        mutation_rate: float | None = None,   # None -> 1 / n_genes
        tournament_size: int = 3,
        elitism: int = 2,
        **kwargs,
    ) -> None:
        """hyperparameter GA.
        """
        super().__init__(**kwargs)
        if selection not in SELECTIONS:
            raise ValueError(f"selection tidak dikenal: {selection}")
        if crossover not in CROSSOVERS:
            raise ValueError(f"crossover tidak dikenal: {crossover}")
        if mutation not in MUTATIONS:
            raise ValueError(f"mutation tidak dikenal: {mutation}")
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

    def extra_params(self) -> dict:
        """Hyperparameter khusus GA, ikut disimpan di RunResult.params"""
        return {
            "population_size": self.population_size,
            "selection": self.selection,
            "crossover": self.crossover,
            "mutation": self.mutation,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "tournament_size": self.tournament_size,
            "elitism": self.elitism,
        }

    def initialize(self) -> None:
        """Generasi 0: populasi acak, lalu hitung fitness tiap individu"""
        self.population = self.random_genomes(self.population_size)
        self.fitness = self.evaluate(self.population)

    def iterate(self) -> None:
        """Satu iterasi: elitisme + seleksi -> crossover -> mutasi -> populasi baru"""
        order = np.argsort(self.fitness)
        elite = order[: self.elitism]

        n_children = self.population_size - self.elitism
        parents = self._select(2 * n_children)
        children = []
        for k in range(n_children):
            a = self.population[parents[2 * k]]
            b = self.population[parents[2 * k + 1]]
            if self.rng.random() < self.crossover_rate:
                child = self._crossover(a, b)
            else:
                child = a.copy()
            children.append(self._mutate(child))

        children = np.array(children)
        children_fitness = self.evaluate(children)
        self.population = np.vstack([self.population[elite], children])
        self.fitness = np.concatenate([self.fitness[elite], children_fitness])

    def _select(self, n: int) -> np.ndarray:
        """Pilih n indeks parent dari populasi (fitness makin kecil makin baik)"""
        size = len(self.population)

        if self.selection == "tournament":
            candidates = self.rng.integers(0, size, size=(n, self.tournament_size))
            winners = np.argmin(self.fitness[candidates], axis=1)
            return candidates[np.arange(n), winners]

        if self.selection == "roulette":
            weight = self.fitness.max() - self.fitness + 1e-9
        else:
            rank = np.empty(size)
            rank[np.argsort(self.fitness)] = np.arange(size)
            weight = size - rank
        return self.rng.choice(size, size=n, p=weight / weight.sum())

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Menggabungkan dua parent menjadi satu anak"""
        n = len(parent_a)

        if self.crossover == "one_point":
            cut = self.rng.integers(1, n)
            return np.concatenate([parent_a[:cut], parent_b[cut:]])

        if self.crossover == "two_point":
            i, j = np.sort(self.rng.choice(np.arange(1, n), size=2, replace=False))
            child = parent_a.copy()
            child[i:j] = parent_b[i:j]
            return child

        take_a = self.rng.random(n) < 0.5
        return np.where(take_a, parent_a, parent_b)

    def _mutate(self, offspring: np.ndarray) -> np.ndarray:
        """Mutasi tiap gen dengan peluang mutation_rate"""
        # default 1/n: rata-rata satu gen termutasi per anak
        rate = self.mutation_rate if self.mutation_rate is not None else 1 / self.n_genes
        hit = np.flatnonzero(self.rng.random(self.n_genes) < rate)

        if self.mutation == "bit_flip":
            counts = self.choice_counts[hit]
            offspring[hit] = (offspring[hit] + self.rng.integers(1, counts)) % counts
        else:
            for i in hit:
                j = self.rng.integers(self.n_genes)
                offspring[i], offspring[j] = offspring[j], offspring[i]
        return offspring

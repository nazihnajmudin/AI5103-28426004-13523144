from __future__ import annotations

import numpy as np

from src.algorithm.base import Algorithm


class ParticleSwarm(Algorithm):
    """Binary PSO (Kennedy & Eberhart, 1997) dengan opsi transfer V-shaped (Mirjalili & Lewis, 2013)
    """

    name = "swarm"

    def __init__(
        self,
        *,
        swarm_size: int = 40,
        inertia: float = 0.7,
        inertia_end: float | None = None,   # None -> inertia konstan
        cognitive: float = 1.5,
        social: float = 1.5,
        v_max: float = 4.0,
        transfer: str = "sigmoid",          # "sigmoid" | "vshape"
        **kwargs,
    ) -> None:
        """hyperparameter BPSO.
        """
        super().__init__(**kwargs)
        if transfer not in ("sigmoid", "vshape"):
            raise ValueError(f"transfer tidak dikenal: {transfer}")
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.inertia_end = inertia_end
        self.cognitive = cognitive
        self.social = social
        self.v_max = v_max
        self.transfer = transfer

    def extra_params(self) -> dict:
        """Hyperparameter khusus BPSO, ikut disimpan di RunResult.params"""
        return {
            "swarm_size": self.swarm_size,
            "inertia": self.inertia,
            "inertia_end": self.inertia_end,
            "cognitive": self.cognitive,
            "social": self.social,
            "v_max": self.v_max,
            "transfer": self.transfer,
        }

    def initialize(self) -> None:
        """Iterasi 0: swarm acak dan jadikan posisi awal sebagai pbest"""
        if np.any(self.choice_counts != 2):
            raise ValueError("BPSO hanya untuk dieset dengan tepat 2 pilihan mesin")
        self.position = self.random_genomes(self.swarm_size)
        self.velocity = self.rng.uniform(-self.v_max, self.v_max, size=self.position.shape)
        self.fitness = self.evaluate(self.position)

        self.pbest = self.position.copy()
        self.pbest_fitness = self.fitness.copy()

    def iterate(self) -> None:
        """Satu iterasi: update kecepatan -> posisi baru -> evaluasi -> update pbest"""
        r1 = self.rng.random(self.velocity.shape)
        r2 = self.rng.random(self.velocity.shape)
        self.velocity = (
            self._inertia() * self.velocity
            + self.cognitive * r1 * (self.pbest - self.position)
            + self.social * r2 * (self.best_genome - self.position)
        )
        np.clip(self.velocity, -self.v_max, self.v_max, out=self.velocity)

        self.position = self._transfer(self.velocity, self.position)
        self.fitness = self.evaluate(self.position)
        improved = self.fitness < self.pbest_fitness
        self.pbest[improved] = self.position[improved]
        self.pbest_fitness[improved] = self.fitness[improved]

    def _inertia(self) -> float:
        """Inertia konstan, atau turun linear inertia -> inertia_end"""
        if self.inertia_end is None:
            return self.inertia

        horizon = self.max_iterations
        if self.max_evaluations is not None:
            horizon = min(horizon, self.max_evaluations // self.swarm_size)
        progress = min(1.0, self.iteration / max(1, horizon))
        return self.inertia - (self.inertia - self.inertia_end) * progress

    def _transfer(self, velocity: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Kecepatan -> posisi biner baru"""
        r = self.rng.random(velocity.shape)
        if self.transfer == "sigmoid":
            prob = 1.0 / (1.0 + np.exp(-velocity))
            return (r < prob).astype(np.int64)

        flip = r < np.abs(np.tanh(velocity))
        return np.where(flip, 1 - position, position)

from __future__ import annotations

import numpy as np

from src.algorithm.base import Algorithm


class ParticleSwarm(Algorithm):
    name = "swarm"

    def __init__(
        self,
        *,
        # TODO: parameters
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # TODO: init parameters

    def extra_params(self) -> dict:
        return {
            # TODO: return dict value
        }

    def initialize(self) -> None:
        # TODO: inisialisasi
        raise NotImplementedError

    def iterate(self) -> None:
        # TODO: iterasi pencarian
        raise NotImplementedError

    # TODO: helper functions

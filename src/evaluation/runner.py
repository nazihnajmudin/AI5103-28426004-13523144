"""Loop percobaan: konfigurasi x instance x seed -> RunResult -> disimpan ke disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.algorithm.base import Algorithm, RunResult
from src.paths import RUNS_DIR
from src.problem import ProblemInstance


def make_seeds(master_seed: int, n_runs: int) -> list[int]:
    """n_runs seed yang diundi dari satu master seed."""
    rng = np.random.default_rng(master_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_runs)]


def run_path(algorithm: str, config: str, instance_key: str, seed: int, runs_dir: Path = RUNS_DIR) -> Path:
    return Path(runs_dir) / algorithm / config / f"instance_{instance_key}" / f"seed_{seed}.json"


def run_experiment(
    configs: dict[str, Algorithm],
    instances: list[ProblemInstance],
    seeds: list[int],
    runs_dir: Path = RUNS_DIR,
) -> None:
    """Jalankan setiap konfigurasi pada semua instance x semua seed, simpan ke disk.
    File hasil yang sudah ada akan ditimpa.

    `configs` = {nama_konfigurasi: objek algoritma}. Objek boleh dipakai ulang
    karena `run()` selalu mereset state-nya.
    """
    for config, algo in configs.items():
        for inst in instances:
            for seed in seeds:
                result = algo.run(inst, seed)
                result.config = config
                result.save(run_path(algo.name, config, inst.key, seed, runs_dir))
        print(f"[{algo.name}] {config:<14} selesai, {len(instances) * len(seeds)} run")


def load_results(algorithm: str | None = None, runs_dir: Path = RUNS_DIR) -> list[RunResult]:
    """Baca hasil run dari disk. `algorithm=None` berarti semua algoritma."""
    runs_dir = Path(runs_dir)
    folder = runs_dir if algorithm is None else runs_dir / algorithm
    results = [RunResult.load(path) for path in sorted(folder.rglob("seed_*.json"))]
    return sorted(results, key=lambda r: (r.algorithm, r.config, r.n_diesets, r.seed))

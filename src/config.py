"""Konfigurasi percobaan dari `configs/experiment.json`: optimum, seed, budget, dan hyperparameter."""

from __future__ import annotations

import json
from pathlib import Path

from src.algorithm import GeneticAlgorithm, ParticleSwarm, RandomSearch
from src.algorithm.base import Algorithm
from src.paths import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "experiment.json"

ALGORITHMS = {
    "random_search": RandomSearch,
    "swarm": ParticleSwarm,
    "genetic": GeneticAlgorithm,
}


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def build_algorithm(algorithm: str, config_name: str, config: dict | None = None) -> Algorithm:
    """Satu objek algoritma: hyperparameter konfigurasi + budget bersama"""
    config = config or load_config()
    params = config["algorithms"][algorithm][config_name]
    return ALGORITHMS[algorithm](**params, **config["budget"])


def build_configs(algorithm: str, config: dict | None = None) -> dict[str, Algorithm]:
    """Semua konfigurasi satu algoritma -> {nama_konfigurasi: objek algoritma}"""
    config = config or load_config()
    return {name: build_algorithm(algorithm, name, config) for name in config["algorithms"][algorithm]}

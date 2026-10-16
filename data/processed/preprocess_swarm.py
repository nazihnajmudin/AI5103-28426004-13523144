"""Preprocessing khusus swarm.

Turunan dari preprocess.py dengan dua perubahan:
  1. kapasitas mesin dikurangi beban dieset fixed (kapasitas residual)
  2. ground truth diambil dari brute-force, nilai paper disimpan sebagai pembanding
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import INSTANCE_SIZES, SRC_DIR, build_instance, load_reference_model

OUT_DIR = Path(__file__).resolve().parent / "instances_swarm"
TOL = 1e-6


def fixed_load_per_machine(raw: dict) -> dict:
    """Total jam kerja dieset fixed per machine_id"""
    load = {}
    for c in raw["model"]["dieset_choices"]:
        if len(c["machine_choices"]) == 1:
            mc = c["machine_choices"][0]
            mid = mc["machine"]["machine_id"]
            load[mid] = load.get(mid, 0.0) + mc["work_required_hours"]
    return load


def to_matrix(instance: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Instance -> cost, work, machine index (n_diesets, 2) dan capacity (n_machines,)"""
    pos = {m["machine_id"]: i for i, m in enumerate(instance["machines"])}
    cost = np.array([[c["total_cost"] for c in d["choices"]] for d in instance["diesets"]])
    work = np.array([[c["work_hours"] for c in d["choices"]] for d in instance["diesets"]])
    mach = np.array([[pos[c["machine_id"]] for c in d["choices"]] for d in instance["diesets"]])
    capacity = np.array([m["capacity_hours"] for m in instance["machines"]])
    return cost, work, mach, capacity


def evaluate_all(genomes: np.ndarray, cost, work, mach, capacity) -> tuple[np.ndarray, np.ndarray]:
    """Cost dan status feasible untuk banyak genome sekaligus"""
    g = genomes.astype(float)
    total = cost[:, 0].sum() + g @ (cost[:, 1] - cost[:, 0])

    feasible = np.ones(len(genomes), dtype=bool)
    for m in range(len(capacity)):
        w0 = np.where(mach[:, 0] == m, work[:, 0], 0.0)
        w1 = np.where(mach[:, 1] == m, work[:, 1], 0.0)
        load = w0.sum() + g @ (w1 - w0)
        feasible &= load <= capacity[m] + TOL
    return total, feasible


def brute_force(instance: dict) -> dict:
    cost, work, mach, capacity = to_matrix(instance)
    n = len(instance["diesets"])
    assert cost.shape[1] == 2, "brute-force di sini hanya untuk 2 pilihan mesin per dieset"

    # baris k = representasi biner dari k -> semua 2^n kombinasi
    genomes = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    total, feasible = evaluate_all(genomes, cost, work, mach, capacity)

    best = int(np.argmin(np.where(feasible, total, np.inf)))
    genome = genomes[best]
    assignment = {
        d["dieset_id"]: d["choices"][int(genome[i])]["machine_id"]
        for i, d in enumerate(instance["diesets"])
    }
    return {
        "optimal_cost": float(total[best]),
        "optimal_genome": genome.tolist(),
        "assignment": assignment,
        "n_feasible": int(feasible.sum()),
        "n_total": int(2**n),
    }


def assignment_to_genome(instance: dict, assignment: dict) -> np.ndarray:
    return np.array([
        next(j for j, c in enumerate(d["choices"]) if c["machine_id"] == assignment[d["dieset_id"]])
        for d in instance["diesets"]
    ])


def build_swarm_instance(n_diesets: int, info: pd.DataFrame) -> dict:
    instance = build_instance(n_diesets, info)
    fixed_load = fixed_load_per_machine(load_reference_model(n_diesets))

    for m in instance["machines"]:
        load = fixed_load.get(m["machine_id"], 0.0)
        m["capacity_hours_total"] = m["capacity_hours"]
        m["fixed_load_hours"] = load
        m["capacity_hours"] = m["capacity_hours_total"] - load

    paper = instance["ground_truth"]
    bf = brute_force(instance)

    # validasi
    matrices = to_matrix(instance)
    bf_cost, bf_ok = evaluate_all(np.array([bf["optimal_genome"]]), *matrices)
    assert abs(bf_cost[0] - bf["optimal_cost"]) < 1e-3, n_diesets
    assert bf_ok[0], n_diesets

    paper_genome = assignment_to_genome(instance, paper["optimal_assignment_dieset_to_machine"])
    paper_cost, paper_ok = evaluate_all(paper_genome[None, :], *matrices)
    assert abs(paper_cost[0] - paper["optimal_cost"]) < 1e-3, n_diesets
    assert paper_ok[0], f"solusi paper tidak feasible pada kapasitas residual (n={n_diesets})"
    assert bf["optimal_cost"] <= paper["optimal_cost"] + 1e-3, n_diesets

    instance["ground_truth"] = {
        "source": "brute_force",
        "optimal_cost": bf["optimal_cost"],
        "optimal_assignment_dieset_to_machine": bf["assignment"],
        "n_feasible": bf["n_feasible"],
        "n_total": bf["n_total"],
        "paper_optimal_cost": paper["optimal_cost"],
        "paper_optimal_bitstring": paper["optimal_bitstring"],
        "paper_assignment_dieset_to_machine": paper["optimal_assignment_dieset_to_machine"],
    }
    return instance


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(SRC_DIR / "info.csv", dtype={"optimal_bitstring": str})

    rows = []
    for n in INSTANCE_SIZES:
        instance = build_swarm_instance(n, info)
        out_path = OUT_DIR / f"instance_{n}_diesets.json"
        with open(out_path, "w") as f:
            json.dump(instance, f, indent=2)

        gt = instance["ground_truth"]
        gap = max(0.0, (gt["paper_optimal_cost"] - gt["optimal_cost"]) / gt["optimal_cost"] * 100)
        caps = "  ".join(f"{m['machine_name']}={m['capacity_hours']:.1f}" for m in instance["machines"])
        print(
            f"[OK] n_diesets={n:2d}  {caps}  feasible={gt['n_feasible']}/{gt['n_total']}  "
            f"optimal={gt['optimal_cost']:.2f}  paper={gt['paper_optimal_cost']:.2f} (+{gap:.3f}%)"
        )

        row = {"n_diesets": n}
        for m in instance["machines"]:
            row[f"capacity_{m['machine_name']}"] = m["capacity_hours"]
            row[f"fixed_load_{m['machine_name']}"] = m["fixed_load_hours"]
        row.update({
            "n_feasible": gt["n_feasible"],
            "n_total": gt["n_total"],
            "feasible_ratio": gt["n_feasible"] / gt["n_total"],
            "optimal_cost": gt["optimal_cost"],
            "paper_optimal_cost": gt["paper_optimal_cost"],
            "paper_gap_pct": gap,
        })
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR.parent / "instances_swarm_summary.csv", index=False)
    print("\nRingkasan ditulis ke instances_swarm_summary.csv")


if __name__ == "__main__":
    main()

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "original" / "src" / "QUBOS"
OUT_DIR = Path(__file__).resolve().parent / "instances"
INSTANCE_SIZES = [3, 9, 13, 16, 18, 19]


def load_reference_model(n_diesets: int) -> dict:
    path = SRC_DIR / f"rounded_qubo_2-machines_{n_diesets}-diesets.json" # n_dieset identik di semua varian QUBO
    with open(path) as f:
        return json.load(f)


def load_optimal(info: pd.DataFrame, n_diesets: int) -> dict:
    row = info[
        (info.n_diesets == n_diesets)
        & (info.qubo_file_name == f"rounded_qubo_2-machines_{n_diesets}-diesets.json")
    ].iloc[0]
    return {"optimal_cost": float(row.optimal_cost), "optimal_bitstring": row.optimal_bitstring}


def choice_record(mc: dict) -> dict:
    machine = mc["machine"]
    return {
        "machine_id": machine["machine_id"],
        "machine_name": machine["machine_name"],
        "work_hours": mc["work_required_hours"],
        # versi bulat ke atas, dipakai SCIP di paper (Schmidbauer et al., Sec. IV-B)
        "work_hours_rounded": math.ceil(mc["work_required_hours"]),
        "work_cost": mc["work_cost"],
        "transport_cost": mc["transport_cost"],
        "total_cost": mc["work_cost"] + mc["transport_cost"],
    }


def fixed_load_per_machine(fixed: list[dict]) -> dict:
    """Total jam kerja dieset fixed di tiap mesin"""
    load = {}
    for c in fixed:
        mc = c["machine_choices"][0]
        mid = mc["machine"]["machine_id"]
        load[mid] = load.get(mid, 0.0) + mc["work_required_hours"]
    return load


def brute_force(instance: dict) -> dict:
    """Coba semua kombinasi genome, ambil solusi feasible termurah (optimum real).
    Dihitung pada model tanpa pembulatan."""
    diesets = instance["diesets"]
    machine_ids = [m["machine_id"] for m in instance["machines"]]
    capacity = np.array([m["capacity_hours"] for m in instance["machines"]])

    cost = np.array([[c["total_cost"] for c in d["choices"]] for d in diesets])
    work = np.array([[c["work_hours"] for c in d["choices"]] for d in diesets])
    machine = np.array([[machine_ids.index(c["machine_id"]) for c in d["choices"]] for d in diesets])

    n = len(diesets)
    genomes = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)
    rows = np.arange(n)
    total_cost = cost[rows, genomes].sum(axis=1)

    feasible = np.ones(len(genomes), dtype=bool)
    for m in range(len(machine_ids)):
        load = np.where(machine[rows, genomes] == m, work[rows, genomes], 0).sum(axis=1)
        feasible &= load <= capacity[m]

    best = np.argmin(np.where(feasible, total_cost, np.inf))
    best_genome = genomes[best]
    assignment = {
        d["dieset_id"]: d["choices"][best_genome[i]]["machine_id"] for i, d in enumerate(diesets)
    }
    return {
        "brute_force_optimal_cost": float(total_cost[best]),
        "brute_force_assignment_dieset_to_machine": assignment,
        "n_feasible": int(feasible.sum()),
        "n_total": len(genomes),
    }


def build_instance(n_diesets: int, info: pd.DataFrame) -> dict:
    raw = load_reference_model(n_diesets)
    dieset_choices = raw["model"]["dieset_choices"]
    varmap = raw["variable_mappings"]

    flexible = [c for c in dieset_choices if len(c["machine_choices"]) > 1]
    fixed = [c for c in dieset_choices if len(c["machine_choices"]) == 1]
    assert len(flexible) == n_diesets, f"expected {n_diesets} flexible diesets, got {len(flexible)}"

    # dieset fixed tetap memakai kapasitas mesin, jadi kapasitas untuk dieset flexible
    # = kapasitas penuh - beban fixed. Versi bulat ke bawah dipakai SCIP di paper.
    fixed_load = fixed_load_per_machine(fixed)

    machines = {}
    diesets = []
    for c in flexible:
        choices = [choice_record(mc) for mc in c["machine_choices"]]
        for mc, rec in zip(c["machine_choices"], choices):
            m = mc["machine"]
            full = m["max_capacity_hours_per_timeslice"]
            used = fixed_load.get(m["machine_id"], 0.0)
            machines.setdefault(
                m["machine_id"],
                {
                    "machine_id": m["machine_id"],
                    "machine_name": m["machine_name"],
                    "capacity_hours": full - used,
                    "capacity_hours_rounded": math.floor(full - used),
                    "max_capacity_hours": full,
                    "fixed_load_hours": used,
                },
            )
        diesets.append(
            {
                "dieset_id": c["dieset_id"],
                "dieset_number": c["dieset_number"],
                "choices": choices,
            }
        )

    optimal = load_optimal(info, n_diesets)

    # decode ground-truth bitstring -> assignment per dieset
    # untuk validasi & referensi solusi optimal siap pakai
    nonslack_bits = optimal["optimal_bitstring"][: raw["num_non_slack_variables"]]
    optimal_assignment = {}
    for i, bit in enumerate(nonslack_bits):
        if bit == "1":
            vm = varmap[i]
            optimal_assignment[vm["dieset_id"]] = vm["machine_id"]

    instance = {
        "n_diesets": n_diesets,
        "n_machines": len(machines),
        "n_fixed_diesets_ignored": len(fixed),
        "machines": sorted(machines.values(), key=lambda m: m["machine_id"]),
        "diesets": diesets,
        "ground_truth": {
            "optimal_cost": optimal["optimal_cost"],
            "optimal_bitstring": optimal["optimal_bitstring"],
            "optimal_assignment_dieset_to_machine": optimal_assignment,
        },
    }
    instance["ground_truth"].update(brute_force(instance))
    return instance


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(SRC_DIR / "info.csv", dtype={"optimal_bitstring": str})

    summary_rows = []
    for n in INSTANCE_SIZES:
        instance = build_instance(n, info)
        out_path = OUT_DIR / f"instance_{n}_diesets.json"
        with open(out_path, "w") as f:
            json.dump(instance, f, indent=2)

        # validasi: fitness dari ground-truth assignment == optimal_cost,
        # dan feasible di model bulat (yang dipakai SCIP)
        cost = 0.0
        load = {m["machine_id"]: 0.0 for m in instance["machines"]}
        for d in instance["diesets"]:
            chosen_machine = instance["ground_truth"]["optimal_assignment_dieset_to_machine"][d["dieset_id"]]
            choice = next(c for c in d["choices"] if c["machine_id"] == chosen_machine)
            cost += choice["total_cost"]
            load[chosen_machine] += choice["work_hours_rounded"]
        assert abs(cost - instance["ground_truth"]["optimal_cost"]) < 1e-3, n
        for m in instance["machines"]:
            assert load[m["machine_id"]] <= m["capacity_hours_rounded"], (n, m)

        gt = instance["ground_truth"]
        assert gt["brute_force_optimal_cost"] <= gt["optimal_cost"] + 1e-6, n
        paper_gap = (gt["optimal_cost"] - gt["brute_force_optimal_cost"]) / gt["brute_force_optimal_cost"]

        print(
            f"[OK] n_diesets={n:2d}  n_machines={instance['n_machines']}  "
            f"fixed_diesets_ignored={instance['n_fixed_diesets_ignored']:2d}  "
            f"optimal_cost={gt['optimal_cost']:.2f}  brute_force={gt['brute_force_optimal_cost']:.2f} "
            f"({paper_gap:+.5%})  -> {out_path.name}"
        )
        machines = {m["machine_name"]: m for m in instance["machines"]}
        summary_rows.append(
            {
                "n_diesets": n,
                "n_machines": instance["n_machines"],
                "n_fixed_diesets_in_source_ignored": instance["n_fixed_diesets_ignored"],
                "n_decision_variables": n * instance["n_machines"],
                "optimal_cost": gt["optimal_cost"],
                "brute_force_optimal_cost": gt["brute_force_optimal_cost"],
                "capacity_GP260": machines["GP260"]["capacity_hours"],
                "capacity_GP640": machines["GP640"]["capacity_hours"],
                "fixed_load_GP260": machines["GP260"]["fixed_load_hours"],
                "n_feasible": gt["n_feasible"],
                "n_total": gt["n_total"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR.parent / "instances_summary.csv", index=False)
    print("\nRingkasan ditulis ke data/processed/instances_summary.csv:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

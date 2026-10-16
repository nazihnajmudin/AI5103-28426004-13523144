import json
from pathlib import Path

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
        "work_cost": mc["work_cost"],
        "transport_cost": mc["transport_cost"],
        "total_cost": mc["work_cost"] + mc["transport_cost"],
    }


def build_instance(n_diesets: int, info: pd.DataFrame) -> dict:
    raw = load_reference_model(n_diesets)
    dieset_choices = raw["model"]["dieset_choices"]
    varmap = raw["variable_mappings"]

    flexible = [c for c in dieset_choices if len(c["machine_choices"]) > 1]
    fixed = [c for c in dieset_choices if len(c["machine_choices"]) == 1]
    assert len(flexible) == n_diesets, f"expected {n_diesets} flexible diesets, got {len(flexible)}"

    machines = {}
    diesets = []
    for c in flexible:
        choices = [choice_record(mc) for mc in c["machine_choices"]]
        for mc, rec in zip(c["machine_choices"], choices):
            m = mc["machine"]
            machines.setdefault(
                m["machine_id"],
                {
                    "machine_id": m["machine_id"],
                    "machine_name": m["machine_name"],
                    "capacity_hours": m["max_capacity_hours_per_timeslice"],
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

    return {
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(SRC_DIR / "info.csv", dtype={"optimal_bitstring": str})

    summary_rows = []
    for n in INSTANCE_SIZES:
        instance = build_instance(n, info)
        out_path = OUT_DIR / f"instance_{n}_diesets.json"
        with open(out_path, "w") as f:
            json.dump(instance, f, indent=2)

        # validasi: fitness dari ground-truth assignment == optimal_cost
        cost = 0.0
        load = {m["machine_id"]: 0.0 for m in instance["machines"]}
        for d in instance["diesets"]:
            chosen_machine = instance["ground_truth"]["optimal_assignment_dieset_to_machine"][d["dieset_id"]]
            choice = next(c for c in d["choices"] if c["machine_id"] == chosen_machine)
            cost += choice["total_cost"]
            load[chosen_machine] += choice["work_hours"]
        assert abs(cost - instance["ground_truth"]["optimal_cost"]) < 1e-3, n
        for m in instance["machines"]:
            assert load[m["machine_id"]] <= m["capacity_hours"] + 1e-6, (n, m)

        print(
            f"[OK] n_diesets={n:2d}  n_machines={instance['n_machines']}  "
            f"fixed_diesets_ignored={instance['n_fixed_diesets_ignored']:2d}  "
            f"optimal_cost={instance['ground_truth']['optimal_cost']:.2f}  -> {out_path.name}"
        )
        summary_rows.append(
            {
                "n_diesets": n,
                "n_machines": instance["n_machines"],
                "n_fixed_diesets_in_source_ignored": instance["n_fixed_diesets_ignored"],
                "n_decision_variables": n * instance["n_machines"],
                "optimal_cost": instance["ground_truth"]["optimal_cost"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR.parent / "instances_summary.csv", index=False)
    print("\nRingkasan ditulis ke opsi_b_data/instances_summary.csv:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

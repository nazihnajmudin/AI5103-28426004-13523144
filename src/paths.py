from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
INSTANCES_DIR = DATA_DIR / "processed" / "instances"
INSTANCES_SUMMARY = DATA_DIR / "processed" / "instances_summary.csv"

RESULTS_DIR = REPO_ROOT / "exp_results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_result_dirs() -> None:
    for d in (RESULTS_DIR, RUNS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

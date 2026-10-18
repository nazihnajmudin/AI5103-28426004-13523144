"""Reproduksi seluruh tabel dan gambar di exp_results/.

Notebook dijalankan berurutan: random_search -> swarm -> genetic -> comparison.
Seed, budget, dan konfigurasi diambil dari configs/experiment.json.

Jalankan dari root repo:
    uv run python experiments/run_all.py                 # semua notebook (sekitar 6 menit)
    uv run python experiments/run_all.py comparison      # hanya notebook tertentu
    uv run python experiments/run_all.py --save          # sekaligus simpan output ke file notebook
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ORDER = ["random_search", "swarm", "genetic", "comparison"]


def run_notebook(name: str, save: bool) -> float:
    path = EXPERIMENTS_DIR / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    started = perf_counter()
    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(EXPERIMENTS_DIR)}})
    client.execute()
    if save:
        nbformat.write(nb, path)
    return perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", help=f"notebook yang dijalankan, default semua: {' '.join(ORDER)}")
    parser.add_argument("--save", action="store_true", help="simpan output sel ke file notebook")
    args = parser.parse_args()

    unknown = set(args.notebooks) - set(ORDER)
    if unknown:
        parser.error(f"notebook tidak dikenal: {', '.join(sorted(unknown))}")
    notebooks = [n for n in ORDER if not args.notebooks or n in args.notebooks]
    for name in notebooks:
        print(f"[run] {name}.ipynb ...", flush=True)
        try:
            elapsed = run_notebook(name, args.save)
        except Exception as exc:
            print(f"[gagal] {name}.ipynb: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"[ok]  {name}.ipynb ({elapsed:.0f} detik)")

    print("\nTabel di exp_results/tables/, gambar di exp_results/figures/")


if __name__ == "__main__":
    main()

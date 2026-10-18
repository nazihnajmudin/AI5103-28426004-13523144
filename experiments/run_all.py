"""Reproduksi seluruh tabel dan gambar di exp_results/.

Notebook dijalankan berurutan: random_search -> swarm -> genetic -> comparison.
Seed, budget, dan konfigurasi diambil dari configs/experiment.json.

Jalankan dari root repo:
    uv run python experiments/run_all.py                 # semua notebook (sekitar 6 menit)
    uv run python experiments/run_all.py comparison      # hanya notebook tertentu
    uv run python experiments/run_all.py --save          # sekaligus simpan output ke file notebook
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ORDER = ["random_search", "swarm", "genetic", "comparison"]


def run_notebook(name: str, save: bool) -> None:
    path = EXPERIMENTS_DIR / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(EXPERIMENTS_DIR)}})
    client.execute()
    if save:
        nbformat.write(nb, path)


def main():
    args = sys.argv[1:]
    save = "--save" in args
    chosen = [a for a in args if a != "--save"]

    for name in ORDER:
        if chosen and name not in chosen:
            continue
        print(f"menjalankan {name}.ipynb ...")
        run_notebook(name, save)
        print(f"selesai {name}.ipynb")

    print("Tabel di exp_results/tables/, gambar di exp_results/figures/")


if __name__ == "__main__":
    main()

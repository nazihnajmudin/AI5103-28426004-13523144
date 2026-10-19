from __future__ import annotations

import numpy as np

from src.algorithm.base import Algorithm

SELECTIONS = ("tournament", "roulette", "rank")
CROSSOVERS = ("one_point", "two_point", "uniform")
MUTATIONS = ("bit_flip", "swap")


class GeneticAlgorithm(Algorithm):
    """GA generasional: seleksi -> crossover -> mutasi, dengan elitisme

    Pemetaan ke persoalan:
      - kromosom / individu = satu alokasi lengkap (vektor biner, 1 gen per dieset)
      - gen ke-i            = mesin untuk dieset i; 0 -> GP260, 1 -> GP640
      - populasi            = kumpulan individu (default 40)
      - fitness             = biaya + penalti kapasitas (base.evaluate), makin kecil makin baik

    Kelas dasar (base.Algorithm) yang menangani evaluasi fitness, pencatatan solusi terbaik,
    dan kriteria berhenti. Kelas ini hanya mengisi initialize() dan iterate().
    """

    name = "genetic"

    def __init__(
        self,
        *,
        population_size: int = 40,
        selection: str = "tournament",
        crossover: str = "uniform",
        mutation: str = "bit_flip",
        crossover_rate: float = 0.9,
        mutation_rate: float | None = None,   # None -> 1 / n_genes
        tournament_size: int = 3,
        elitism: int = 2,
        **kwargs,
    ) -> None:
        """Simpan hyperparameter GA.

        population_size jumlah individu per generasi
        selection       cara memilih parent: "tournament" | "roulette" | "rank"
        crossover       cara menggabungkan dua parent: "one_point" | "two_point" | "uniform"
        mutation        cara memutasi anak: "bit_flip" | "swap"
        crossover_rate  p_c, peluang sepasang parent di-crossover (kalau tidak, anak = salinan parent a)
        mutation_rate   p_m, peluang tiap gen dimutasi; None -> 1/n (rata-rata 1 gen per anak)
        tournament_size jumlah kandidat per turnamen (hanya untuk tournament selection)
        elitism         jumlah individu terbaik yang langsung lolos ke generasi berikutnya
        kwargs          budget (max_evaluations, max_iterations, patience) diteruskan ke base
        """
        super().__init__(**kwargs)
        if selection not in SELECTIONS:
            raise ValueError(f"selection tidak dikenal: {selection}")
        if crossover not in CROSSOVERS:
            raise ValueError(f"crossover tidak dikenal: {crossover}")
        if mutation not in MUTATIONS:
            raise ValueError(f"mutation tidak dikenal: {mutation}")
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

    def extra_params(self) -> dict:
        """Hyperparameter khusus GA, ikut disimpan di RunResult.params"""
        return {
            "population_size": self.population_size,
            "selection": self.selection,
            "crossover": self.crossover,
            "mutation": self.mutation,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "tournament_size": self.tournament_size,
            "elitism": self.elitism,
        }

    def initialize(self) -> None:
        """Generasi 0: populasi acak, lalu hitung fitness tiap individu"""
        # (population_size, n_dieset) gen acak; evaluate() juga mencatat solusi terbaik
        self.population = self.random_genomes(self.population_size)
        self.fitness = self.evaluate(self.population)

    def iterate(self) -> None:
        """Satu generasi: elitisme + seleksi -> crossover -> mutasi -> populasi baru"""
        # elitisme: individu terbaik langsung masuk generasi berikutnya
        # (argsort menaik, jadi indeks pertama = fitness terkecil = terbaik)
        order = np.argsort(self.fitness)
        elite = order[: self.elitism]

        # sisa slot populasi diisi anak; setiap anak butuh 2 parent
        # contoh default: 40 - 2 elit = 38 anak -> 76 parent dipilih (boleh terpilih berulang)
        n_children = self.population_size - self.elitism
        parents = self._select(2 * n_children)
        children = []
        for k in range(n_children):
            # pasangan parent ke-k: parents[2k] dan parents[2k+1]
            a = self.population[parents[2 * k]]
            b = self.population[parents[2 * k + 1]]
            # crossover dengan peluang crossover_rate, kalau tidak anak = salinan parent a
            if self.rng.random() < self.crossover_rate:
                child = self._crossover(a, b)
            else:
                child = a.copy()
            # setiap anak tetap dimutasi, baik hasil crossover maupun salinan
            children.append(self._mutate(child))

        # hanya anak yang dievaluasi (38 evaluasi per generasi); fitness elit sudah diketahui
        children = np.array(children)
        children_fitness = self.evaluate(children)
        # generasi baru = elit + anak; populasi lama selain elit dibuang (generasional)
        self.population = np.vstack([self.population[elite], children])
        self.fitness = np.concatenate([self.fitness[elite], children_fitness])

    def _select(self, n: int) -> np.ndarray:
        """Pilih n indeks parent dari populasi (fitness makin kecil makin baik)"""
        size = len(self.population)

        if self.selection == "tournament":
            # ambil tournament_size kandidat acak -> pemenang = fitness terkecil
            # ukuran 3: individu terburuk hampir tidak pernah menang -> tekanan seleksi kuat
            candidates = self.rng.integers(0, size, size=(n, self.tournament_size))
            winners = np.argmin(self.fitness[candidates], axis=1)
            return candidates[np.arange(n), winners]

        if self.selection == "roulette":
            # peluang -> selisih ke fitness terburuk
            # minimasi: makin kecil fitness, makin besar bobot f_max - f;
            # 1e-9 supaya bobot tidak semuanya 0 kalau fitness seluruh populasi sama
            weight = self.fitness.max() - self.fitness + 1e-9
        else:
            # rank: terbaik dapat bobot size, terburuk dapat bobot 1
            # yang dipakai urutan, bukan besar selisih fitness -> tekanan seleksi lebih lembut
            rank = np.empty(size)
            rank[np.argsort(self.fitness)] = np.arange(size)
            weight = size - rank
        # undi n parent sesuai bobot (dengan pengembalian, satu individu bisa terpilih berkali-kali)
        return self.rng.choice(size, size=n, p=weight / weight.sum())

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Menggabungkan dua parent menjadi satu anak"""
        n = len(parent_a)

        if self.crossover == "one_point":
            # potong di satu titik: gen [0, cut) dari a, gen [cut, n) dari b
            cut = self.rng.integers(1, n)
            return np.concatenate([parent_a[:cut], parent_b[cut:]])

        if self.crossover == "two_point":
            # dua titik potong i < j: segmen tengah [i, j) dari b, sisanya dari a
            i, j = np.sort(self.rng.choice(np.arange(1, n), size=2, replace=False))
            child = parent_a.copy()
            child[i:j] = parent_b[i:j]
            return child

        # uniform: tiap gen diambil dari salah satu parent dengan peluang 0,5
        # cocok di sini karena urutan dieset di kromosom tidak punya makna
        take_a = self.rng.random(n) < 0.5
        return np.where(take_a, parent_a, parent_b)

    def _mutate(self, offspring: np.ndarray) -> np.ndarray:
        """Mutasi tiap gen dengan peluang mutation_rate"""
        # default 1/n: rata-rata satu gen termutasi per anak
        rate = self.mutation_rate if self.mutation_rate is not None else 1 / self.n_genes
        hit = np.flatnonzero(self.rng.random(self.n_genes) < rate)

        if self.mutation == "bit_flip":
            # move dieset ke pilihan mesin lain
            # dengan 2 pilihan: (gen + 1) % 2 -> 0 jadi 1, 1 jadi 0
            counts = self.choice_counts[hit]
            offspring[hit] = (offspring[hit] + self.rng.integers(1, counts)) % counts
        else:
            # swap: tukar nilai gen terpilih dengan gen acak lain
            # jumlah dieset di tiap mesin tidak berubah, hanya dieset mana yang di mesin mana
            for i in hit:
                j = self.rng.integers(self.n_genes)
                offspring[i], offspring[j] = offspring[j], offspring[i]
        return offspring

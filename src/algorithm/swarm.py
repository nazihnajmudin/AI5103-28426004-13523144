from __future__ import annotations

import numpy as np

from src.algorithm.base import Algorithm


class ParticleSwarm(Algorithm):
    """Binary PSO (Kennedy & Eberhart, 1997) dengan opsi transfer V-shaped (Mirjalili & Lewis, 2013)

    Pemetaan ke persoalan:
      - satu partikel   = satu alokasi lengkap (vektor biner, 1 bit per dieset)
      - satu dimensi d  = satu dieset; bit 0 -> GP260, bit 1 -> GP640
      - kecepatan v[d]  = kecenderungan dieset d dipindah ke GP640 (positif) atau GP260 (negatif)
      - pbest           = posisi terbaik yang pernah dicapai tiap partikel
      - gbest           = posisi terbaik seluruh swarm (self.best_genome, dikelola base.evaluate())

    Kelas dasar (base.Algorithm) yang menangani evaluasi fitness, pencatatan solusi terbaik,
    dan kriteria berhenti. Kelas ini hanya mengisi initialize() dan iterate().
    """

    name = "swarm"

    def __init__(
        self,
        *,
        swarm_size: int = 40,
        inertia: float = 0.7,
        inertia_end: float | None = None,   # None -> inertia konstan
        cognitive: float = 1.5,
        social: float = 1.5,
        v_max: float = 4.0,
        transfer: str = "sigmoid",          # "sigmoid" | "vshape"
        **kwargs,
    ) -> None:
        """Simpan hyperparameter BPSO.

        swarm_size  jumlah partikel (= jumlah evaluasi per iterasi)
        inertia     w, bobot kecepatan lama; kalau inertia_end diisi, ini nilai awalnya
        inertia_end nilai akhir w saat inertia turun linear
        cognitive   c1, tarikan ke pbest (pengalaman partikel sendiri)
        social      c2, tarikan ke gbest (pengalaman swarm)
        v_max       batas absolut kecepatan
        transfer    fungsi yang mengubah kecepatan menjadi bit
        kwargs      budget (max_evaluations, max_iterations, patience) diteruskan ke base
        """
        super().__init__(**kwargs)
        if transfer not in ("sigmoid", "vshape"):
            raise ValueError(f"transfer tidak dikenal: {transfer}")
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.inertia_end = inertia_end
        self.cognitive = cognitive
        self.social = social
        self.v_max = v_max
        self.transfer = transfer

    def extra_params(self) -> dict:
        """Hyperparameter khusus BPSO, ikut disimpan di RunResult.params"""
        return {
            "swarm_size": self.swarm_size,
            "inertia": self.inertia,
            "inertia_end": self.inertia_end,
            "cognitive": self.cognitive,
            "social": self.social,
            "v_max": self.v_max,
            "transfer": self.transfer,
        }

    def initialize(self) -> None:
        """Iterasi 0: bangkitkan swarm acak dan jadikan posisi awal sebagai pbest"""
        # BPSO bekerja di ruang biner, jadi tiap dieset harus punya tepat 2 pilihan mesin
        if np.any(self.choice_counts != 2):
            raise ValueError("BPSO hanya untuk dieset dengan tepat 2 pilihan mesin")

        # posisi: (swarm_size, n_dieset) bit acak; kecepatan: bilangan acak di [-v_max, v_max]
        self.position = self.random_genomes(self.swarm_size)
        self.velocity = self.rng.uniform(-self.v_max, self.v_max, size=self.position.shape)
        # evaluate() juga memperbarui gbest (self.best_genome) dan menghitung jumlah evaluasi
        self.fitness = self.evaluate(self.position)

        # belum ada riwayat, jadi posisi terbaik tiap partikel = posisi awalnya
        self.pbest = self.position.copy()
        self.pbest_fitness = self.fitness.copy()

    def iterate(self) -> None:
        """Satu iterasi: update kecepatan -> posisi baru -> evaluasi -> update pbest"""
        # r1, r2 acak per partikel per dimensi, supaya tarikan ke pbest/gbest bervariasi
        r1 = self.rng.random(self.velocity.shape)
        r2 = self.rng.random(self.velocity.shape)

        # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        # (pbest - x) dan (gbest - x) bernilai -1, 0, atau 1 karena posisinya biner:
        #   1  -> bit ini sebaiknya jadi 1, kecepatan didorong ke positif
        #   -1 -> bit ini sebaiknya jadi 0, kecepatan didorong ke negatif
        #   0  -> sudah sama, hanya inertia yang bekerja
        # gbest = self.best_genome, sudah di-update oleh evaluate()
        self.velocity = (
            self._inertia() * self.velocity
            + self.cognitive * r1 * (self.pbest - self.position)
            + self.social * r2 * (self.best_genome - self.position)
        )
        # batasi kecepatan supaya peluang bit tidak menempel di 0 atau 1 (swarm tetap eksplorasi)
        np.clip(self.velocity, -self.v_max, self.v_max, out=self.velocity)

        # kecepatan -> posisi biner baru, lalu hitung fitness-nya (sekaligus update gbest)
        self.position = self._transfer(self.velocity, self.position)
        self.fitness = self.evaluate(self.position)

        # pbest hanya diganti oleh partikel yang posisi barunya lebih baik (fitness lebih kecil)
        improved = self.fitness < self.pbest_fitness
        self.pbest[improved] = self.position[improved]
        self.pbest_fitness[improved] = self.fitness[improved]

    def _inertia(self) -> float:
        """Inertia konstan, atau turun linear inertia -> inertia_end"""
        if self.inertia_end is None:
            return self.inertia

        # horizon mengikuti batas yang paling cepat tercapai
        # (budget 5.000 evaluasi / 40 partikel = 125 iterasi, jauh sebelum max_iterations)
        horizon = self.max_iterations
        if self.max_evaluations is not None:
            horizon = min(horizon, self.max_evaluations // self.swarm_size)
        # progress 0 di awal -> 1 di akhir; w turun dari inertia ke inertia_end
        # w besar di awal = eksplorasi, w kecil di akhir = eksploitasi
        progress = min(1.0, self.iteration / max(1, horizon))
        return self.inertia - (self.inertia - self.inertia_end) * progress

    def _transfer(self, velocity: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Kecepatan -> posisi biner baru"""
        r = self.rng.random(velocity.shape)
        if self.transfer == "sigmoid":
            # S-shaped: S(v) = 1 / (1 + e^-v) = peluang bit bernilai 1
            # v besar positif -> hampir pasti 1 (GP640), v besar negatif -> hampir pasti 0 (GP260),
            # v = 0 -> peluang 0,5 (acak)
            prob = 1.0 / (1.0 + np.exp(-velocity))
            return (r < prob).astype(np.int64)

        # V-shaped: T(v) = |tanh(v)| = peluang bit dibalik (0 <-> 1)
        # |v| besar -> kemungkinan besar dibalik, v = 0 -> bit tetap (partikel "diam")
        flip = r < np.abs(np.tanh(velocity))
        return np.where(flip, 1 - position, position)

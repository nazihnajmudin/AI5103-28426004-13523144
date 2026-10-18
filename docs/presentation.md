# Materi Presentasi

Isi 12 slide (batas maksimal spesifikasi) untuk presentasi 12 menit. Urutan slide mengikuti bagian presentasi di spesifikasi: persoalan dan data → rancangan solusi → demo → hasil dan analisis → kesimpulan.

Sumber angka: `README.md`, `docs/results/`, dan `data/processed/instances_summary.csv`. Simpangan baku, nilai terburuk, dan uji Wilcoxon di slide 9 dihitung ulang dari runner kelompok dengan seed dan budget yang sama, dan rata-ratanya identik dengan `docs/results/comparison.md`. Path gambar relatif terhadap folder `docs/`.

| Bagian | Durasi | Slide |
|---|---|---|
| Pembuka | – | 1 |
| Persoalan dan data | 2 menit | 2 |
| Rancangan solusi | 3 menit | 3–6 |
| Demo langsung | 3 menit | 10 |
| Hasil dan analisis | 3 menit | 7–9 |
| Kesimpulan dan keterbatasan | 1 menit | 11 |
| Penutup dan tanya jawab | 5 menit | 12 |

Demo ditaruh setelah hasil supaya alur cerita runtut. Kalau dosen meminta urutan sesuai tabel spesifikasi, cukup pindahkan slide 10 ke antara slide 6 dan 7.

---

## Slide 1 — Cover

**Judul:** Penerapan Genetic Algorithm dan Binary PSO untuk Alokasi Dieset pada Press Shop BMW

**Subjudul:** Problem Solving Agent pada dataset Path Matters

- AI5103 Fundamental of AI & Data Science, Semester 1 2026/2027
- [Nama – NIM], [Nama – NIM]
- [Tanggal presentasi]

**Visual:** `img/case-illustration.png`

---

## Slide 2 — Dataset & Industrial Problem

**Persoalan**
- Press shop BMW: setiap dieset (alat cetak body mobil) harus dipasang di tepat satu mesin press.
- Tujuan: meminimalkan biaya produksi + transportasi, tanpa melebihi kapasitas jam mesin.
- Menarik karena datanya nyata dari industri, dan paper sumbernya menyelesaikannya dengan komputer kuantum (D-Wave, IBM). Kita uji apakah metaheuristik klasik bisa menyamai.

**Data**
- Path Matters, Schmidbauer et al., IEEE QCE 2025 (arXiv:2504.16607).
- Zenodo DOI 10.5281/zenodo.16269046 (record 16269046).
- 6 instance, 2 mesin (GP260 dan GP640). Dieset fixed mengurangi kapasitas mesin.

| Dieset flexible | 3 | 9 | 13 | 16 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|
| Ruang pencarian $2^n$ | 8 | 512 | 8.192 | 65.536 | 262.144 | 524.288 |
| Kombinasi feasible | 1 | 179 | 2.951 | 27.821 | 114.795 | 222.128 |

**Visual:** `img/case-illustration.png`

---

## Slide 3 — Solution-Problem Representation

- **State/solusi:** vektor biner sepanjang jumlah dieset flexible. `gene[i] = 0` berarti GP260, `gene[i] = 1` berarti GP640.
- Representasi yang sama dipakai sebagai kromosom GA, posisi partikel BPSO, dan sampel random search.
- **Assignment constraint** otomatis terpenuhi, karena setiap gen memilih tepat satu mesin.
- **Capacity constraint** ditangani dengan fungsi penalti:

$$f = \sum_i c_{i,\text{gene}[i]} + \lambda \sum_m \max(0, \text{load}_m - h_m)^2 + \lambda \cdot \mathbb{1}[\text{ada overload}]$$

- $\lambda$ = total biaya terbesar yang mungkin. Akibatnya, solusi infeasible selalu lebih buruk daripada solusi feasible mana pun, dan solusi akhir yang dilaporkan selalu valid.

**Visual:** `img/solution-representation.png`

---

## Slide 4 — QUBO Solver, Baseline, and Experimental Setup

**Pendekatan di paper:** masalah dikodekan sebagai QUBO dengan variabel slack. Akibatnya, 19 dieset menjadi 60 qubit. Paper membandingkan D-Wave, LR-QAOA, dan Simulated Annealing terhadap random reference.

**Nilai acuan (optimum)**

| Instance | Optimum SCIP (paper) | Optimum brute-force |
|---:|---:|---:|
| 16 | 5.803.504,82 | 5.801.121,21 |
| 18 | 5.963.876,62 | 5.962.911,99 |
| 19 | 6.115.761,14 | 6.112.960,59 |

- SCIP memakai kapasitas dan beban yang dibulatkan. Brute-force memakai nilai asli, dan dipakai sebagai acuan gap.
- Instance 3, 9, dan 13 memiliki optimum yang sama.

**Baseline:** random search, karena setara dengan random reference di paper dan menunjukkan kontribusi mekanisme pencarian.

**Setting yang adil:** semua algoritma diberi 5.000 evaluasi fitness, 30 run per instance, dan seed yang sama (master seed 2026).

**Metrik:** gap $= (\text{hasil} - \text{optimum}) / \text{optimum} \times 100\%$ (terbaik, rata-rata, simpangan baku, terburuk), % feasible, % optimum ditemukan, dan waktu.

---

## Slide 5 — Algorithm: GA

| Proses | Default | Variasi yang dibandingkan | Alasan |
|---|---|---|---|
| Seleksi | tournament (ukuran 3) | roulette, rank | tekanan seleksi berbeda |
| Crossover | uniform, $p_c = 0{,}9$ | one-point, two-point | gen tidak punya urutan yang bermakna |
| Mutasi | bit-flip, $p_m = 1/n$ | swap | bit-flip = memindah satu dieset ke mesin lain |
| Populasi | 40, elitisme 2 | 20, 80 | 40 = ukuran swarm BPSO |

- Siklus per generasi: seleksi → crossover → mutasi → gabung dengan 2 individu elit.
- Penetapan parameter: nilai default umum di literatur, lalu analisis sensitivitas dengan mengubah satu faktor per konfigurasi (8 konfigurasi).

**Visual:** diagram alur GA.

---

## Slide 6 — Algorithm: BPSO

- **Alasan memilih:** variabel keputusannya biner, dan BPSO adalah PSO untuk ruang biner. Representasi dan fitness-nya sama dengan GA, sehingga perbandingannya adil.
- **Pemetaan:** satu partikel = satu alokasi, satu dimensi = satu dieset, kecepatan = kecenderungan dieset dipindah ke GP640.

$$v_{id} \leftarrow w\,v_{id} + c_1 r_1 (p_{id} - x_{id}) + c_2 r_2 (g_d - x_{id})$$

| Transfer | Rumus | Aturan |
|---|---|---|
| Sigmoid | $1/(1+e^{-v})$ | bit = 1 jika $r < S(v)$ |
| V-shaped | $\lvert\tanh(v)\rvert$ | bit dibalik jika $r < T(v)$ |

- Default: sigmoid, $w = 0{,}7$, $c_1 = c_2 = 1{,}5$, $v_{max} = 4$ (peluang bit tetap di 0,018–0,982), swarm 40.
- Sensitivitas: inertia linear 0,9 → 0,4, V-shaped, swarm 20 dan 80, $v_{max} = 6$ (7 konfigurasi).

**Visual:** diagram alur BPSO (update kecepatan → transfer → evaluasi → update pbest/gbest).

---

## Slide 7 — Result and Observation: GA

| Konfigurasi | Gap rata-rata 6 instance (%) | Optimum ditemukan (%) |
|---|---:|---:|
| `sel_rank` (terbaik) | 0,0119 | 57,2 |
| `pop_80` | 0,0128 | 60,0 |
| `default` | 0,0172 | 51,7 |
| `mut_swap` | 0,0262 | 43,3 |
| `pop_20` (terburuk) | 0,0270 | 43,9 |

**Analisis sensitivitas**
- Seleksi paling berpengaruh: rank selection menurunkan gap dari 0,0172% menjadi 0,0119%.
- Swap mutation terburuk, karena tidak mengubah jumlah dieset di tiap mesin.
- Populasi besar lebih baik: 80 > 40 > 20.
- Semua 1.440 run feasible, dengan gap di bawah 1%.

**Visual:** `img/results/genetic_configs_gap.png`

---

## Slide 8 — Result and Observation: BPSO

| Konfigurasi | Gap rata-rata 6 instance (%) | Optimum ditemukan (%) |
|---|---:|---:|
| `sigmoid_s20` (terbaik) | 0,0102 | 60,6 |
| `sigmoid_v6` | 0,0111 | 63,9 |
| `sigmoid` (default) | 0,0149 | 52,8 |
| `vshape` (terburuk) | 0,0177 | 50,0 |

**Analisis sensitivitas**
- Dengan inertia konstan, sigmoid lebih baik daripada V-shaped. Inertia linear membuat keduanya hampir setara.
- Swarm kecil lebih baik: dengan budget yang sama, swarm 20 mendapat 249 iterasi, sedangkan swarm 80 hanya 62 iterasi.
- Selisih antar konfigurasi lebih kecil daripada sebaran antar seed.
- Semua 1.260 run feasible, dengan gap di bawah 1%.

**Visual:** `img/results/swarm_configs_gap.png`

---

## Slide 9 — Result and Observation: Comparison

Konfigurasi terbaik tiap algoritma, 30 run per instance. Gap dalam %, ditulis sebagai rata-rata ± simpangan baku (terburuk). Gap terbaik GA dan BPSO = 0 di semua instance.

| n | GA `sel_rank` | BPSO `sigmoid_s20` | Random `batch_20` | Solusi SCIP | Wilcoxon GA vs BPSO (p) |
|---:|---|---|---|---:|---:|
| 13 | 0,0072 ± 0,0184 (0,0598) | 0,0024 ± 0,0096 (0,0508) | 0,0077 ± 0,0110 (0,0424) | 0 | 0,62 |
| 16 | 0,0220 ± 0,0195 (0,0710) | 0,0180 ± 0,0206 (0,0694) | 0,0697 ± 0,0238 (0,1191) | 0,0411 | 0,33 |
| 18 | 0,0084 ± 0,0085 (0,0360) | 0,0101 ± 0,0104 (0,0490) | 0,0412 ± 0,0208 (0,1027) | 0,0162 | 0,62 |
| 19 | 0,0338 ± 0,0161 (0,0737) | 0,0308 ± 0,0189 (0,0737) | 0,0948 ± 0,0439 (0,2063) | 0,0458 | 0,66 |

**Analisis**
- GA vs BPSO **tidak berbeda signifikan** (Wilcoxon rank-sum, p > 0,3 di semua instance).
- GA dan BPSO **signifikan lebih baik** daripada random search mulai n = 16 (p < 10⁻⁸). Gap random 3–5 kali lebih besar.
- Gap rata-rata GA dan BPSO lebih kecil daripada solusi SCIP di instance 16, 18, dan 19.
- BPSO 4,5 kali lebih cepat (0,018 detik per run, dibanding GA 0,081 detik per run), karena tidak ada operator per anak.

**Visual:** `img/results/comparison_convergence.png`

---

## Slide 10 — Demo

Menjalankan program pada **instance 19 dieset** dengan seed dan budget yang sama untuk ketiga algoritma:

1. Kurva konvergensi GA, BPSO, dan random search yang bergerak selama pencarian.
2. Solusi akhir: alokasi dieset ke GP260/GP640 dan beban tiap mesin terhadap kapasitasnya.
3. Tabel perbandingan cost dan gap ketiga algoritma terhadap optimum.

Cadangan: video demo maksimal 3 menit.

---

## Slide 11 — Conclusion and Limitations

**Kesimpulan**
- GA dan BPSO selalu menghasilkan solusi valid dengan gap < 0,08%, dan menemukan optimum minimal sekali di setiap instance.
- Kualitas GA dan BPSO tidak berbeda signifikan. BPSO lebih cepat.
- Keduanya signifikan lebih baik daripada random search mulai n = 16.

**Keterbatasan**
- Instance kecil (maksimal 19 dieset, 2 mesin), sehingga brute-force masih bisa dihitung.
- Sensitivitas hanya satu faktor per konfigurasi, tanpa interaksi antar parameter.
- Pada n = 16 dan 18, budget 5.000 evaluasi belum cukup untuk konvergen penuh.

**Ide perbaikan:** instance lebih besar atau lebih banyak mesin, inisialisasi dari solusi greedy, local search, dan baseline kedua (Hill Climbing atau Simulated Annealing).

---

## Slide 12 — Thank You

**Terima kasih**

- Repositori: github.com/nazihnajmudin/AI5103-28426004-13523144
- Referensi utama:
  1. Schmidbauer et al., *Path Matters: Industrial Data Meet Quantum Optimization*, IEEE QCE 2025.
  2. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*, 1989.
  3. Kennedy & Eberhart, *A Discrete Binary Version of the Particle Swarm Algorithm*, 1997.
  4. Mirjalili & Lewis, *S-shaped versus V-shaped Transfer Functions for Binary PSO*, 2013.

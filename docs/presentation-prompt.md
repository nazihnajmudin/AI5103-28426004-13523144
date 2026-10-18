# Prompt Pembuatan PPT

Lampirkan `docs/presentation.md` beserta gambar yang disebut di dalamnya (`docs/img/case-illustration.png`, `docs/img/solution-representation.png`, dan isi `docs/img/results/`), lalu tempel prompt di bawah ini ke tool pembuat presentasi.

---

```text
Buatkan presentasi PowerPoint (16:9) dari file presentation.md yang saya lampirkan.

Struktur:
- Tepat 12 slide (batas maksimal tugas), urutan dan judul sesuai heading "Slide 1" sampai "Slide 12":
  1. Cover
  2. Dataset & Industrial Problem
  3. Solution-Problem Representation
  4. QUBO Solver, Baseline, and Experimental Setup
  5. Algorithm: GA
  6. Algorithm: BPSO
  7. Result and Observation: GA
  8. Result and Observation: BPSO
  9. Result and Observation: Comparison
  10. Demo
  11. Conclusion and Limitations
  12. Thank You
- Judul slide dalam bahasa Inggris seperti di atas, isi slide dalam bahasa Indonesia.
- Abaikan tabel durasi di awal file; itu panduan untuk presenter, bukan slide.

Aturan isi:
- Gunakan hanya isi dari file. Jangan menambah angka, klaim, atau referensi baru.
- Semua angka, nilai p, dan nama konfigurasi (misalnya sel_rank, sigmoid_s20) disalin persis, termasuk koma desimal.
- Maksimal 5 poin per slide, setiap poin satu baris. Kalau isi di file lebih panjang, ringkas tanpa mengubah maknanya, dan pindahkan detailnya ke speaker notes.
- Tabel di file tetap ditampilkan sebagai tabel. Pada slide 9, kolom BPSO diberi penekanan warna, dan kolom nilai p tetap ditampilkan.
- Rumus ditampilkan sebagai persamaan (equation), bukan teks biasa.
- Pada baris "Visual:", sisipkan gambar dengan nama file tersebut. Kalau gambarnya tidak tersedia, buat placeholder bertuliskan nama filenya.
- Slide 5 dan 6: buat diagram alur sederhana sesuai deskripsi "Visual" di file.
- Slide 1: biarkan placeholder [Nama – NIM] dan [Tanggal presentasi] apa adanya.

Gaya:
- Akademik dan bersih: latar putih, satu warna aksen (biru tua), font sans-serif.
- Tanpa animasi, tanpa clip-art, tanpa ikon dekoratif.
- Setiap slide diberi speaker notes 2–3 kalimat yang menjelaskan poin utamanya.
```

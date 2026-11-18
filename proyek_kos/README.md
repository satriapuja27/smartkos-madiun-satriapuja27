# SmartKos Madiun.AI — Portal Sistem Cerdas Terpadu

Aplikasi web berbasis **Django** yang mengintegrasikan tiga pendekatan utama Sistem Cerdas untuk membantu mahasiswa Politeknik Negeri Madiun (PNM) dalam menganalisis tarif, mencari informasi, dan menentukan kelayakan tempat kos:

1. **Bagian A — Aplikasi Prediksi Harga ML:** Memprediksi estimasi tarif sewa kos bulanan menggunakan algoritma **Linear Regression** (`model_kos.joblib`) yang dilatih dari dataset `kos_madiun.csv`, lengkap dengan slider interaktif, preset cepat, serta grafik perbandingan rata-rata tarif kamar ber-AC dan Non-AC.
2. **Bagian B — Chatbot FAQ Berbasis RAG Mini:** Asisten tanya-jawab interaktif yang dilengkapi **10 entri basis pengetahuan FAQ Kos Madiun** dan mesin *Retrieval-Augmented Generation* (RAG) mini berbasis **TF-IDF & Cosine Similarity** (`scikit-learn`) lengkap dengan tampilan referensi dokumen terambil.
3. **Bagian C — Sistem Pakar & Fuzzy Web:** Sistem pendukung keputusan berbasis **Fuzzy Inference System (FIS)** pada variabel anggaran serta jarak ke kampus yang dipadukan dengan **9 aturan inferensi berantai (*Forward Chaining*)** dan fasilitas jejak penalaran (*Rule Trace*).

Proyek ini disusun untuk memenuhi tugas **Mini Proyek — Pertemuan 14 (Deployment)** pada mata kuliah **Sistem Cerdas (TI24317)**, Program Studi Teknologi Informasi, Politeknik Negeri Madiun.

---

## Sumber & Lisensi Dataset
- **Dataset Utama:** `kos_madiun.csv` — Dataset praktikum mata kuliah Sistem Cerdas (TI24317), Program Studi Teknologi Informasi, Politeknik Negeri Madiun (Dosen Pengampu: Muhammad Syaeful Fajar, S.Pd. Gr., M.Kom.).
- **Basis Pengetahuan FAQ & Aturan Pakar:** Disusun secara mandiri berdasarkan observasi tarif, fasilitas, dan aturan kos mahasiswa di sekitar Kampus PNM Madiun.
- **Lisensi:** Penggunaan terbatas untuk keperluan akademik, praktikum, dan pembelajaran mahasiswa.

---

## Cara Menjalankan Aplikasi di Lokal
1. Buka terminal (**PowerShell** atau **CMD**) di dalam folder `proyek_kos`.
2. Pastikan seluruh *library* terinstal:
   ```powershell
   py -m pip install django pandas scikit-learn joblib
   ```
3. Jalankan server lokal Django:
   ```powershell
   py manage.py runserver
   ```
4. Buka browser dan akses alamat `http://127.0.0.1:8000/`.

---

## Pengembang & Kontribusi Kerja
- **Nama:** Satria Puja Asmara Abadi
- **Peran & Pembagian Tugas (Proyek Individu):** Merancang dan membangun keseluruhan sistem secara mandiri, mencakup pelatihan model regresi linear (`model_kos.joblib`), penyusunan basis pengetahuan 10 FAQ dan mesin RAG TF-IDF, perancangan aturan Fuzzy & Sistem Pakar, pengembangan *backend* Django (`views.py`), pembuatan antarmuka web modern (`index.html`), serta penyusunan dokumentasi proyek (`README.md`).
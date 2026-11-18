import json
import os
import joblib
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. PEMUATAN MODEL ML & DATASET (BAGIAN A)
# ==============================================================================
MODEL_PATH = os.path.join(settings.BASE_DIR, "model_kos.joblib")
DATASET_PATH = os.path.join(settings.BASE_DIR, "kos_madiun.csv")

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    df_train = pd.read_csv(DATASET_PATH)
    df_train["ac_num"] = df_train["ac"].map({"ya": 1, "tidak": 0})
    model = LinearRegression().fit(df_train[["jarak_km", "rating", "ac_num"]], df_train["harga"])
    joblib.dump(model, MODEL_PATH)

def ambil_data_grafik():
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    rata_ac = df.groupby("ac")["harga"].mean()
    harga_ac = round(float(rata_ac.get("ya", 0)))
    harga_non_ac = round(float(rata_ac.get("tidak", 0)))
    rata_total = round(float(df["harga"].mean()))
    maks = max(harga_ac, harga_non_ac, 1)
    return {
        "ac_ribu": f"{harga_ac:,}".replace(",", "."),
        "ac_rupiah": f"{harga_ac * 1000:,}".replace(",", "."),
        "ac_persen": round((harga_ac / maks) * 100),
        "non_ac_ribu": f"{harga_non_ac:,}".replace(",", "."),
        "non_ac_rupiah": f"{harga_non_ac * 1000:,}".replace(",", "."),
        "non_ac_persen": round((harga_non_ac / maks) * 100),
        "rata_total_rupiah": f"{rata_total * 1000:,}".replace(",", "."),
        "rata_total_nilai": rata_total,
        "total_data": len(df),
    }

# ==============================================================================
# 2. BASIS PENGETAHUAN FAQ & RAG MINI ENGINE (BAGIAN B - 10 ENTRI)
# ==============================================================================
FAQ_ENTRIES = [
    {
        "id": "FAQ-01",
        "kategori": "Tarif & Biaya",
        "pertanyaan": "Berapa kisaran harga sewa kos bulanan di sekitar kampus Madiun?",
        "jawaban": "Berdasarkan dataset kos Madiun, tarif sewa berkisar antara Rp 450.000 hingga Rp 1.600.000 per bulan. Rata-rata kos Non-AC sekitar Rp 1.047.000/bulan, sedangkan kos ber-AC rata-rata Rp 1.068.000/bulan tergantung jarak ke kampus dan rating fasilitas.",
    },
    {
        "id": "FAQ-02",
        "kategori": "Fasilitas AC",
        "pertanyaan": "Apakah kamar kos dengan fasilitas AC jauh lebih mahal daripada Non-AC?",
        "jawaban": "Selisih rata-rata harga kos AC dan Non-AC relatif terjangkau, namun pemakaian listrik pada kos ber-AC umumnya menggunakan token listrik mandiri (prabayar) per kamar sehingga perlu menyiapkan anggaran listrik bulanan sekitar Rp 100.000–Rp 200.000.",
    },
    {
        "id": "FAQ-03",
        "kategori": "Lokasi & Jarak",
        "pertanyaan": "Area mana saja yang paling dekat dengan kampus Politeknik Negeri Madiun (PNM)?",
        "jawaban": "Area favorit mahasiswa berjarak 0,3 km hingga 1,5 km dari Kampus 1 & Kampus 2 PNM meliputi kawasan Jalan Serayu, Kelurahan Pandean, Taman, dan Demangan karena akses jalan kaki atau sepeda motor sangat dekat.",
    },
    {
        "id": "FAQ-04",
        "kategori": "Sistem Pembayaran",
        "pertanyaan": "Bagaimana sistem pembayaran sewa kos di Madiun, bulanan atau tahunan?",
        "jawaban": "Mayoritas kos mahasiswa di Madiun menerapkan pembayaran per bulan atau per 3 bulan (triwulan). Beberapa kos eksklusif meminta uang jaminan (deposit) kunci/kerusakan sebesar Rp 100.000–Rp 300.000 yang dikembalikan saat masa sewa berakhir.",
    },
    {
        "id": "FAQ-05",
        "kategori": "Fasilitas Kamar",
        "pertanyaan": "Fasilitas standar apa saja yang biasanya sudah termasuk dalam harga sewa kos?",
        "jawaban": "Fasilitas standar umumnya mencakup tempat tidur (kasur & bantal), lemari pakaian, meja belajar, akses Wi-Fi bersama, serta air bersih (PDAM/sumur pompa). Kamar mandi dalam dan water heater biasanya tersedia pada kos dengan rating di atas 4.2.",
    },
    {
        "id": "FAQ-06",
        "kategori": "Aturan & Jam Malam",
        "pertanyaan": "Apakah kos di Madiun memberlakukan aturan jam malam bagi penghuni?",
        "jawaban": "Sebagian besar kos putri dan kos yang menyatu dengan rumah pemilik memberlakukan jam malam pukul 22.00 atau 23.00 WIB. Jika sering pulang malam karena kegiatan praktikum atau organisasi, pilih kos dengan akses gerbang 24 jam dan kunci mandiri.",
    },
    {
        "id": "FAQ-07",
        "kategori": "Keamanan & Parkir",
        "pertanyaan": "Bagaimana keamanan parkir sepeda motor di kos sekitar kampus?",
        "jawaban": "Kos dengan rating di atas 4.0 umumnya sudah menyediakan area parkir motor beratap (kanopi) di dalam pagar terkunci dan dilengkapi kamera CCTV 24 jam.",
    },
    {
        "id": "FAQ-08",
        "kategori": "Tips Survei",
        "pertanyaan": "Apa saja yang wajib dicek saat survei lokasi kos secara langsung?",
        "jawaban": "Periksa kelancaran aliran air keran, sinyal seluler dan kecepatan Wi-Fi di dalam kamar, sirkulasi udara/ventilasi (penting untuk cuaca Madiun yang cukup terik), kondisi kunci pintu/jendela, serta jarak ke warung makan dan laundry.",
    },
    {
        "id": "FAQ-09",
        "kategori": "Metodologi ML",
        "pertanyaan": "Bagaimana cara kerja fitur Prediksi Harga pada aplikasi SmartKos Madiun ini?",
        "jawaban": "Fitur prediksi menggunakan algoritma Regresi Linear (LinearRegression) yang dilatih menggunakan 30 data kos Madiun berdasarkan tiga variabel utama: jarak ke kampus (km), rating fasilitas (1–5), dan ketersediaan AC.",
    },
    {
        "id": "FAQ-10",
        "kategori": "Panduan Rekomendasi",
        "pertanyaan": "Bagaimana memilih kos terbaik jika budget terbatas di bawah Rp 750.000?",
        "jawaban": "Gunakan tab Sistem Pakar & Fuzzy pada aplikasi ini. Secara umum, pilih kos Non-AC dengan jarak 1,5–2,5 km dari kampus yang memiliki ventilasi silang baik serta kipas angin agar biaya sewa dan listrik tetap hemat.",
    },
]

# Membangun indeks vektor TF-IDF sekali saat server aktif untuk RAG Mini
_korpus_faq = [
    f"{item['kategori']} {item['pertanyaan']} {item['jawaban']}" for item in FAQ_ENTRIES
]
_vectorizer = TfidfVectorizer()
_matriks_tfidf = _vectorizer.fit_transform(_korpus_faq)

def jalankan_rag_chatbot(pertanyaan_user):
    vektor_tanya = _vectorizer.transform([pertanyaan_user])
    skor_kemiripan = cosine_similarity(vektor_tanya, _matriks_tfidf).flatten()
    indeks_terurut = skor_kemiripan.argsort()[::-1]

    konteks_terambil = []
    for idx in indeks_terurut[:2]:
        skor = float(skor_kemiripan[idx])
        if skor > 0.06:
            entri = FAQ_ENTRIES[idx]
            konteks_terambil.append({
                "id": entri["id"],
                "kategori": entri["kategori"],
                "pertanyaan": entri["pertanyaan"],
                "jawaban": entri["jawaban"],
                "skor_persen": round(skor * 100, 1),
            })

    if not konteks_terambil:
        jawaban_final = (
            "Maaf, pertanyaan tersebut belum tercakup secara spesifik dalam 10 basis pengetahuan FAQ Kos Madiun. "
            "Coba tanyakan seputar kisaran harga sewa, fasilitas AC, jarak ke kampus PNM, aturan jam malam, parkir, atau tips survei kos."
        )
    elif len(konteks_terambil) == 1 or konteks_terambil[0]["skor_persen"] >= 35.0:
        jawaban_final = konteks_terambil[0]["jawaban"]
    else:
        jawaban_final = (
            f"{konteks_terambil[0]['jawaban']} "
            f"Selain itu sebagai informasi tambahan terkait {konteks_terambil[1]['kategori'].lower()}: "
            f"{konteks_terambil[1]['jawaban']}"
        )

    return jawaban_final, konteks_terambil

# ==============================================================================
# 3. SISTEM PAKAR & FUZZY INFERENCE SYSTEM (BAGIAN C - FIS + 9 ATURAN BERANTAI)
# ==============================================================================
def hitung_fuzzy_dan_pakar(budget_ribu, jarak_km, fokus_belajar, toleransi_suhu):
    # A. Fuzzifikasi Variabel 1: Budget (dalam ribu Rupiah)
    if budget_ribu <= 650:
        mu_budget_hemat = 1.0
    elif budget_ribu < 900:
        mu_budget_hemat = round((900 - budget_ribu) / 250, 2)
    else:
        mu_budget_hemat = 0.0

    if 650 < budget_ribu < 900:
        mu_budget_sedang = round((budget_ribu - 650) / 250, 2)
    elif 900 <= budget_ribu <= 1150:
        mu_budget_sedang = 1.0
    elif 1150 < budget_ribu < 1400:
        mu_budget_sedang = round((1400 - budget_ribu) / 250, 2)
    else:
        mu_budget_sedang = 0.0

    if budget_ribu <= 1150:
        mu_budget_tinggi = 0.0
    elif budget_ribu < 1400:
        mu_budget_tinggi = round((budget_ribu - 1150) / 250, 2)
    else:
        mu_budget_tinggi = 1.0

    # B. Fuzzifikasi Variabel 2: Jarak ke Kampus (km)
    if jarak_km <= 1.0:
        mu_jarak_dekat = 1.0
    elif jarak_km < 2.2:
        mu_jarak_dekat = round((2.2 - jarak_km) / 1.2, 2)
    else:
        mu_jarak_dekat = 0.0

    if 1.0 < jarak_km < 2.0:
        mu_jarak_sedang = round((jarak_km - 1.0) / 1.0, 2)
    elif 2.0 <= jarak_km <= 2.8:
        mu_jarak_sedang = 1.0
    elif 2.8 < jarak_km < 3.8:
        mu_jarak_sedang = round((3.8 - jarak_km) / 1.0, 2)
    else:
        mu_jarak_sedang = 0.0

    if jarak_km <= 2.8:
        mu_jarak_jauh = 0.0
    elif jarak_km < 3.8:
        mu_jarak_jauh = round((jarak_km - 2.8) / 1.0, 2)
    else:
        mu_jarak_jauh = 1.0

    # C. Evaluasi 9 Aturan Inferensi Berantai (Forward Chaining + Fuzzy)
    jejak_aturan = []
    bobot_skor = []

    # Aturan R1 - R6 (Kombinasi Fuzzy Budget x Jarak)
    rules_fis = [
        ("R1", min(mu_budget_hemat, mu_jarak_dekat), 86, "Kos Ekonomis Strategis (Hemat & Dekat Kampus)"),
        ("R2", min(mu_budget_hemat, mu_jarak_sedang), 76, "Kos Ekonomis Menengah"),
        ("R3", min(mu_budget_hemat, mu_jarak_jauh), 62, "Kos Hemat Area Pinggiran (Perlu Transportasi)"),
        ("R4", min(mu_budget_sedang, mu_jarak_dekat), 91, "Kos Standar Favorit Mahasiswa"),
        ("R5", min(mu_budget_sedang, mu_jarak_sedang), 80, "Kos Nyaman Standar"),
        ("R6", min(mu_budget_tinggi, mu_jarak_dekat), 96, "Kos Eksklusif Dekat Kampus"),
    ]

    kategori_terbaik = "Kos Standar Rekomendasi"
    alfa_tertinggi = -1.0

    for kode, alfa, nilai_konsekuen, label_kat in rules_fis:
        if alfa > 0:
            bobot_skor.append((alfa, nilai_konsekuen))
            jejak_aturan.append(
                f"[{kode}] Derajat aktif α = {alfa:.2f} → Mengarah ke '{label_kat}' (Nilai {nilai_konsekuen})"
            )
            if alfa > alfa_tertinggi:
                alfa_tertinggi = alfa
                kategori_terbaik = label_kat

    # Jika berada di kombinasi budget tinggi/sedang & jarak jauh (fallback rule)
    if not bobot_skor:
        bobot_skor.append((1.0, 72))
        kategori_terbaik = "Kos Tenang Area Jauh Kampus"
        jejak_aturan.append("[R6b] Kombinasi jarak jauh dengan budget menengah/tinggi → Skor dasar 72")

    # Defuzzifikasi Weighted Average (Sugeno Orde-0)
    skor_dasar = sum(a * z for a, z in bobot_skor) / sum(a for a, _ in bobot_skor)

    # Aturan Berantai R7, R8, R9 (Fakta Kondisi Penghuni)
    saran_fasilitas = []
    bonus_skor = 0

    if toleransi_suhu == "gerah" and budget_ribu >= 850:
        bonus_skor += 3
        saran_fasilitas.append("Wajib pilih kamar dengan fasilitas AC (budget mencukupi)")
        jejak_aturan.append("[R7 - Aturan Berantai] Sensitif suhu panas + budget ≥ Rp 850 rb → Wajib AC (+3 poin)")
    elif toleransi_suhu == "gerah" and budget_ribu < 850:
        bonus_skor -= 2
        saran_fasilitas.append("Pilih kamar Non-AC dengan ventilasi silang & kipas angin (AC melebihi budget)")
        jejak_aturan.append("[R8 - Aturan Berantai] Sensitif suhu panas + budget hemat → Optimalkan ventilasi & kipas (-2 poin)")
    else:
        saran_fasilitas.append("Kamar Non-AC sudah nyaman dan lebih hemat listrik bulanan")
        jejak_aturan.append("[R8b - Aturan Berantai] Toleransi suhu normal → Kamar Non-AC direkomendasikan")

    if fokus_belajar == "tinggi":
        bonus_skor += 2
        saran_fasilitas.append("Prioritaskan kamar mandi dalam & lingkungan tenang jauh dari jalan utama")
        jejak_aturan.append("[R9 - Aturan Berantai] Kebutuhan fokus belajar tinggi → Prioritas kamar tenang & Wi-Fi stabil (+2 poin)")
    else:
        saran_fasilitas.append("Kos reguler dengan akses dekat pusat kuliner & area komunal cocok untuk aktivitasmu")
        jejak_aturan.append("[R9b - Aturan Berantai] Kebutuhan belajar fleksibel → Kos reguler dekat akses umum sesuai")

    skor_akhir = min(100, max(0, round(skor_dasar + bonus_skor)))

    if skor_akhir >= 85:
        status_kelayakan = "Sangat Direkomendasikan"
    elif skor_akhir >= 70:
        status_kelayakan = "Layak & Sesuai"
    else:
        status_kelayakan = "Perlu Penyesuaian Budget/Jarak"

    return {
        "skor_akhir": skor_akhir,
        "status_kelayakan": status_kelayakan,
        "kategori_terbaik": kategori_terbaik,
        "saran_fasilitas": saran_fasilitas,
        "jejak_aturan": jejak_aturan,
        "fuzzy_budget": {
            "hemat": round(mu_budget_hemat * 100),
            "sedang": round(mu_budget_sedang * 100),
            "tinggi": round(mu_budget_tinggi * 100),
        },
        "fuzzy_jarak": {
            "dekat": round(mu_jarak_dekat * 100),
            "sedang": round(mu_jarak_sedang * 100),
            "jauh": round(mu_jarak_jauh * 100),
        },
    }

# ==============================================================================
# 4. VIEW UTAMA DJANGO (MENANGANI KETIGA MODUL DALAM SATU HALAMAN)
# ==============================================================================
def index(request):
    grafik = ambil_data_grafik()
    active_tab = "ml"

    # State default Bagian A (ML)
    ml_state = {
        "hasil_ribu": None,
        "hasil_rupiah": None,
        "selisih_info": None,
        "jarak": "1.2",
        "rating": "4.3",
        "ac": "ya",
        "error": None,
    }

    # State default Bagian B (Chatbot RAG)
    chat_history = [
        {
            "pengirim": "bot",
            "teks": "Halo! Saya Asisten SmartKos Madiun berbasis RAG Mini (10 entri FAQ). Silakan tanya seputar harga kos, fasilitas AC, jarak ke PNM, jam malam, atau tips survei!",
            "referensi": [],
        }
    ]

    # State default Bagian C (Sistem Pakar & Fuzzy)
    pakar_input = {
        "budget": "850",
        "jarak": "1.2",
        "fokus": "tinggi",
        "suhu": "gerah",
    }
    pakar_hasil = hitung_fuzzy_dan_pakar(850, 1.2, "tinggi", "gerah")

    if request.method == "POST":
        aksi = request.POST.get("aksi", "prediksi")

        # Memulihkan riwayat chat jika dikirim dari form
        riwayat_raw = request.POST.get("riwayat_chat_json")
        if riwayat_raw:
            try:
                parsed = json.loads(riwayat_raw)
                if isinstance(parsed, list) and len(parsed) > 0:
                    chat_history = parsed
            except json.JSONDecodeError:
                pass

        if aksi == "prediksi":
            active_tab = "ml"
            try:
                jarak = float(request.POST.get("jarak"))
                rating = float(request.POST.get("rating"))
                ac = request.POST.get("ac")

                data_baru = pd.DataFrame({
                    "jarak_km": [jarak],
                    "rating": [rating],
                    "ac_num": [1 if ac == "ya" else 0],
                })

                prediksi = model.predict(data_baru)[0]
                harga_bulat = max(0, round(prediksi))

                selisih_info = ""
                if grafik:
                    beda = harga_bulat - grafik["rata_total_nilai"]
                    if beda > 15:
                        selisih_info = f"Rp {beda:,} ribu lebih tinggi dari rata-rata pasar Madiun".replace(",", ".")
                    elif beda < -15:
                        selisih_info = f"Rp {abs(beda):,} ribu lebih hemat dari rata-rata pasar Madiun".replace(",", ".")
                    else:
                        selisih_info = "Sesuai dengan rata-rata harga kos di Madiun"

                ml_state.update({
                    "hasil_ribu": f"{harga_bulat:,}".replace(",", "."),
                    "hasil_rupiah": f"{harga_bulat * 1000:,}".replace(",", "."),
                    "selisih_info": selisih_info,
                    "jarak": str(jarak),
                    "rating": str(rating),
                    "ac": ac,
                })
            except ValueError:
                ml_state["error"] = "Pastikan input jarak dan rating berupa angka yang valid."

        elif aksi == "chat":
            active_tab = "chatbot"
            pertanyaan_user = request.POST.get("pertanyaan", "").strip()
            if pertanyaan_user:
                chat_history.append({"pengirim": "user", "teks": pertanyaan_user, "referensi": []})
                jawaban_bot, referensi = jalankan_rag_chatbot(pertanyaan_user)
                chat_history.append({"pengirim": "bot", "teks": jawaban_bot, "referensi": referensi})

        elif aksi == "pakar":
            active_tab = "pakar"
            try:
                budget = float(request.POST.get("budget", 850))
                jarak_p = float(request.POST.get("jarak_pakar", 1.2))
                fokus = request.POST.get("fokus", "tinggi")
                suhu = request.POST.get("suhu", "gerah")

                pakar_input = {
                    "budget": str(int(budget)),
                    "jarak": str(jarak_p),
                    "fokus": fokus,
                    "suhu": suhu,
                }
                pakar_hasil = hitung_fuzzy_dan_pakar(budget, jarak_p, fokus, suhu)
            except ValueError:
                pass

    context = {
        "active_tab": active_tab,
        "grafik": grafik,
        "ml": ml_state,
        "chat_history": chat_history,
        "chat_history_json": json.dumps(chat_history),
        "faq_count": len(FAQ_ENTRIES),
        "pakar_input": pakar_input,
        "pakar": pakar_hasil,
    }
    return render(request, "prediksi/index.html", context)
import os
import random
import re
import pypdf  # Menggunakan pypdf pengganti fitz
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ==========================================
# PENGATURAN PROGRAM
# ==========================================

FOLDER_DOKUMEN = "dokumen"
FOLDER_HASIL = "hasil"

UKURAN_CHUNK = [500, 1000, 1500]
OVERLAP = 0.15

MODEL = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

os.makedirs(FOLDER_HASIL, exist_ok=True)


# ==========================================
# FUNGSI 1: MEMBACA PDF (Menggunakan pypdf)
# ==========================================

def baca_pdf(path):
    teks = ""
    try:
        reader = pypdf.PdfReader(path)
        for halaman in reader.pages:
            teks_halaman = halaman.extract_text()
            if teks_halaman:
                teks += teks_halaman + "\n"
    except Exception as e:
        print(f"Error membaca PDF dengan pypdf: {e}")
        
    return teks


# ==========================================
# FUNGSI 2: MEMOTONG TEKS (CHUNKING)
# ==========================================

def potong(teks, ukuran, overlap=0.15):
    """
    ukuran  : jumlah karakter setiap chunk
    overlap : persentase teks yang diulang
    """

    if not 0 <= overlap < 1:
        raise ValueError("Overlap harus antara 0 dan 1.")

    if ukuran <= 0:
        raise ValueError("Ukuran chunk harus lebih dari 0.")

    langkah = max(1, int(ukuran * (1 - overlap)))

    chunks = []

    for awal in range(0, len(teks), langkah):
        akhir = awal + ukuran
        bagian = teks[awal:akhir].strip()

        if bagian:
            chunks.append(bagian)

        if akhir >= len(teks):
            break

    return chunks


# ==========================================
# FUNGSI 3: MEMUAT SEMUA PDF
# ==========================================

def muat_dokumen():
    semua_dokumen = []

    if not os.path.exists(FOLDER_DOKUMEN):
        print("Folder dokumen tidak ditemukan!")
        return semua_dokumen

    files = [
        f for f in os.listdir(FOLDER_DOKUMEN)
        if f.lower().endswith(".pdf")
    ]

    if not files:
        print("Tidak ada file PDF dalam folder dokumen.")
        return semua_dokumen

    for nama_file in files:
        path = os.path.join(FOLDER_DOKUMEN, nama_file)

        try:
            teks = baca_pdf(path)

            if teks.strip():
                semua_dokumen.append({
                    "nama_file": nama_file,
                    "teks": teks
                })

                print(
                    f"Berhasil membaca {nama_file}: "
                    f"{len(teks)} karakter"
                )
            else:
                print(f"Tidak ada teks yang terbaca: {nama_file}")

        except Exception as e:
            print(f"Gagal membaca {nama_file}: {e}")

    return semua_dokumen


# ==========================================
# FUNGSI 4: MEMBUAT KOLEKSI CHUNK
# ==========================================

def buat_koleksi(dokumen, ukuran, overlap=0.15):
    koleksi = []

    for item in dokumen:
        chunks = potong(
            item["teks"],
            ukuran,
            overlap
        )

        for nomor, isi in enumerate(chunks):
            koleksi.append({
                "file": item["nama_file"],
                "nomor": nomor + 1,
                "teks": isi
            })

        print(
            f"{item['nama_file']} | "
            f"Chunk ukuran {ukuran}: {len(chunks)}"
        )

    return koleksi


# ==========================================
# FUNGSI 5: MEMBANGUN INDEKS FAISS
# ==========================================

def bangun_indeks(koleksi, nama_indeks):
    if not koleksi:
        raise ValueError("Koleksi chunk kosong.")

    teks_chunks = [
        item["teks"] for item in koleksi
    ]

    embeddings = MODEL.encode(
        teks_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimensi = embeddings.shape[1]

    indeks = faiss.IndexFlatIP(dimensi)
    indeks.add(embeddings)

    # Simpan indeks
    path_indeks = os.path.join(
        FOLDER_HASIL,
        f"{nama_indeks}.faiss"
    )

    faiss.write_index(indeks, path_indeks)

    # Simpan metadata chunk
    path_metadata = os.path.join(
        FOLDER_HASIL,
        f"{nama_indeks}.npy"
    )

    np.save(
        path_metadata,
        np.array(koleksi, dtype=object),
        allow_pickle=True
    )

    print(f"Indeks berhasil dibuat: {nama_indeks}")
    print(f"Jumlah vektor: {indeks.ntotal}")

    return indeks, koleksi


# ==========================================
# FUNGSI 6: QUERY / PENCARIAN SEMANTIK
# ==========================================

def query(pertanyaan, indeks, koleksi, k=3):
    embedding_query = MODEL.encode(
        [pertanyaan],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embedding_query = np.asarray(
        embedding_query,
        dtype="float32"
    )

    jumlah = min(k, indeks.ntotal)

    skor, posisi = indeks.search(
        embedding_query,
        jumlah
    )

    hasil = []

    for i, pos in enumerate(posisi[0]):
        if pos < 0:
            continue

        hasil.append({
            "peringkat": i + 1,
            "skor": float(skor[0][i]),
            "file": koleksi[pos]["file"],
            "nomor_chunk": koleksi[pos]["nomor"],
            "teks": koleksi[pos]["teks"]
        })

    return hasil


# ==========================================
# FUNGSI 7: MENGUKUR HIT@K
# ==========================================

def hit_at_k(daftar_pertanyaan, indeks, koleksi, k):
    berhasil = 0

    for data in daftar_pertanyaan:
        hasil = query(
            data["pertanyaan"],
            indeks,
            koleksi,
            k
        )

        kata_kunci = [
            kata.lower()
            for kata in data["kata_kunci"]
        ]

        ditemukan = False

        for item in hasil:
            teks = item["teks"].lower()

            # Semua kata kunci harus ditemukan
            # dalam minimal satu chunk yang sama.
            if all(
                kata in teks for kata in kata_kunci
            ):
                ditemukan = True
                break

        if ditemukan:
            berhasil += 1

    total = len(daftar_pertanyaan)

    if total == 0:
        return 0

    return berhasil / total * 100


# ==========================================
# LAB 1: EKSPERIMEN CHUNKING
# ==========================================

def lab_1(dokumen):
    print("\n========== LAB 1 ==========")

    if not dokumen:
        return

    teks = dokumen[0]["teks"]

    print("\n500 karakter pertama:")
    print(teks[:500])

    for ukuran in UKURAN_CHUNK:
        chunks = potong(
            teks,
            ukuran,
            OVERLAP
        )

        print("\n--------------------------")
        print(f"Ukuran chunk: {ukuran}")
        print(f"Jumlah chunk: {len(chunks)}")

        # Tampilkan 3 chunk acak
        sampel = random.sample(
            chunks,
            min(3, len(chunks))
        )

        for nomor, isi in enumerate(sampel, 1):
            print(f"\nContoh chunk {nomor}:")
            print(isi)


# ==========================================
# LAB 2: MEMBANGUN INDEKS DAN PENCARIAN
# ==========================================

def lab_2(dokumen):
    print("\n========== LAB 2 ==========")

    ukuran = 500

    koleksi = buat_koleksi(
        dokumen,
        ukuran,
        OVERLAP
    )

    indeks, koleksi = bangun_indeks(
        koleksi,
        "indeks_500"
    )

    print("\nMasukkan pertanyaan.")
    print("Ketik 'keluar' untuk berhenti.")

    while True:
        pertanyaan = input("\nPertanyaan: ")

        if pertanyaan.lower() == "keluar":
            break

        hasil = query(
            pertanyaan,
            indeks,
            koleksi,
            k=3
        )

        print("\nHasil pencarian:")

        for item in hasil:
            print("\nPeringkat:", item["peringkat"])
            print("Skor:", round(item["skor"], 4))
            print("File:", item["file"])
            print("Chunk:", item["nomor_chunk"])
            print("Isi:", item["teks"][:500])


# ==========================================
# LAB 3: EVALUASI HIT@K
# ==========================================

def lab_3(dokumen):
    print("\n========== LAB 3 ==========")

    pertanyaan_uji = [
        {
            "pertanyaan": "Apa yang dimaksud dengan gizi seimbang?",
            "kata_kunci": ["gizi", "seimbang", "makronutrien"]
        },
        {
            "pertanyaan": "Apa fungsi utama karbohidrat bagi tubuh?",
            "kata_kunci": ["karbohidrat", "energi", "biji-bijian"]
        },
        {
            "pertanyaan": "Dari mana sumber protein nabati?",
            "kata_kunci": ["protein", "tahu", "tempe"]
        },
        {
            "pertanyaan": "Apa manfaat lemak sehat seperti omega-3?",
            "kata_kunci": ["lemak", "omega-3", "alpukat"]
        },
        {
            "pertanyaan": "Apa peran mikronutrien dalam tubuh?",
            "kata_kunci": ["mikronutrien", "vitamin", "mineral"]
        },
        {
            "pertanyaan": "Mengapa serat sangat penting bagi pencernaan?",
            "kata_kunci": ["serat", "pencernaan", "gula darah"]
        },
        {
            "pertanyaan": "Di mana saja kita bisa menemukan sumber serat?",
            "kata_kunci": ["buah-buahan", "sayuran", "kacang-kacangan"]
        },
        {
            "pertanyaan": "Berapa persentase air dalam tubuh manusia dewasa?",
            "kata_kunci": ["air", "60%", "tubuh manusia"]
        },
        {
            "pertanyaan": "Berapa anjuran konsumsi air putih per hari?",
            "kata_kunci": ["8 gelas", "cairan", "air putih"]
        },
        {
            "pertanyaan": "Apa dampak negatif dari makanan olahan berlebih?",
            "kata_kunci": ["makanan olahan", "junk food", "obesitas"]
        },
        {
            "pertanyaan": "Bagaimana cara menjaga kesehatan jantung?",
            "kata_kunci": ["jantung", "serat", "pola makan"]
        },
        {
            "pertanyaan": "Apa fungsi utama vitamin bagi tubuh?",
            "kata_kunci": ["vitamin", "kekebalan tubuh", "metabolisme"]
        },
        {
            "pertanyaan": "Apa akibat kekurangan mineral bagi kesehatan?",
            "kata_kunci": ["mineral", "tulang", "metabolisme"]
        },
        {
            "pertanyaan": "Bagaimana tips menghindari gula tambahan berlebih?",
            "kata_kunci": ["gula tambahan", "pola makan", "makanan"]
        },
        {
            "pertanyaan": "Apa manfaat menjaga hidrasi tubuh dengan baik?",
            "kata_kunci": ["hidrasi", "suhu tubuh", "transportasi"]
        }
    ]

    hasil_evaluasi = {}

    for ukuran in [500, 1000]:
        print(f"\nMembuat koleksi ukuran {ukuran}...")
        koleksi = buat_koleksi(dokumen, ukuran, OVERLAP)
        indeks, koleksi = bangun_indeks(koleksi, f"indeks_{ukuran}")
        hasil_evaluasi[ukuran] = {}

        for k in [1, 3, 5]:
            nilai = hit_at_k(pertanyaan_uji, indeks, koleksi, k)
            hasil_evaluasi[ukuran][k] = nilai
            print(f"Chunk {ukuran} | Hit@{k}: {nilai:.2f}%")

    print("\n========== TABEL PERBANDINGAN ==========")
    print("Ukuran Chunk | Hit@1 | Hit@3 | Hit@5")
    for ukuran in [500, 1000]:
        print(
            f"{ukuran:12} | "
            f"{hasil_evaluasi[ukuran][1]:5.2f}% | "
            f"{hasil_evaluasi[ukuran][3]:5.2f}% | "
            f"{hasil_evaluasi[ukuran][5]:5.2f}%"
        )
    # Tampilkan perbandingan
    print("\n========== TABEL PERBANDINGAN ==========")

    print("Ukuran Chunk | Hit@1 | Hit@3 | Hit@5")

    for ukuran in [500, 1000]:
        print(
            f"{ukuran:12} | "
            f"{hasil_evaluasi[ukuran][1]:5.2f}% | "
            f"{hasil_evaluasi[ukuran][3]:5.2f}% | "
            f"{hasil_evaluasi[ukuran][5]:5.2f}%"
        )


# ==========================================
# MENU UTAMA
# ==========================================

def main():
    print("================================")
    print("    PROGRAM PRAKTIKUM RAG")
    print("================================")

    dokumen = muat_dokumen()

    if not dokumen:
        print("Tambahkan PDF terlebih dahulu.")
        return

    while True:
        print("\nPILIH LAB:")
        print("1. Lab 1 - Eksperimen Chunking")
        print("2. Lab 2 - Bangun Indeks dan Cari")
        print("3. Lab 3 - Evaluasi Hit@k")
        print("0. Keluar")

        pilihan = input("\nMasukkan pilihan: ")

        if pilihan == "1":
            lab_1(dokumen)

        elif pilihan == "2":
            lab_2(dokumen)

        elif pilihan == "3":
            lab_3(dokumen)

        elif pilihan == "0":
            print("Program selesai.")
            break

        else:
            print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()
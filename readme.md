# 📚 Lab RAG (Retrieval-Augmented Generation)

Proyek praktikum ini adalah implementasi sistem *Retrieval-Augmented Generation* (RAG) berbasis Python yang berjalan secara lokal menggunakan pustaka `pypdf`, `sentence-transformers`, dan `FAISS`.

## 🛠️ Prasyarat & Dependensi
Proyek ini dirancang untuk berjalan di lingkungan virtual Python (`venv`) dengan pustaka berikut:
- `pypdf` (Pustaka murni Python untuk membaca file PDF)
- `sentence-transformers` (Untuk membuat *embedding* teks)
- `faiss-cpu` (Untuk pencarian kesamaan vektor yang cepat)
- `numpy` (Operasi numerik)
- `fonttools` (Opsional, untuk menangani pengkodean font PDF khusus)

## 📁 Struktur Direktori
```text
lab_RAG/
│
├── dokumen/            # Tempat meletakkan file PDF materi (materi_1.pdf, materi_2.pdf, materi_3.pdf)
├── hasil/              # Tempat penyimpanan indeks FAISS dan metadata (.faiss dan .npy)
├── venv/               # Direktori lingkungan virtual Python
├── main.py             # Script utama program praktikum RAG
└── requirements.txt    # Daftar dependensi pustaka
```

## 🚀 Cara Menjalankan Program

1. **Aktifkan Virtual Environment (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Pastikan File PDF Tersedia:**
   Letakkan file PDF materi Anda (misalnya `materi_1.pdf`, `materi_2.pdf`, dan `materi_3.pdf`) ke dalam folder `dokumen/`.

3. **Jalankan Program Utama:**
   ```powershell
   python main.py
   ```

## 🧪 Menu Praktikum
- **Lab 1:** Eksperimen Chunking (Memecah dokumen dengan berbagai ukuran).
- **Lab 2:** Bangun Indeks dan Cari (Membuat indeks FAISS dan melakukan *semantic search*).
- **Lab 3:** Evaluasi Hit@K (Menguji kinerja pencarian menggunakan pertanyaan uji dengan metrik $k = 1, 3, 5$).
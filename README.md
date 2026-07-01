# 🏢 ArsipPintar: Automated Enterprise Knowledge Base Powered by Local SLM

Selamat datang di **ArsipPintar**! 🚀
ArsipPintar adalah aplikasi *Knowledge Base* tingkat perusahaan berbasis **RAG (Retrieval-Augmented Generation)** yang berjalan **100% lokal, offline, dan gratis (Rp 0 biaya API)**. 

Aplikasi ini dirancang sebagai solusi cerdas untuk menganalisis dokumen internal yang sensitif (seperti SOP, berkas hukum, atau laporan keuangan) tanpa risiko kebocoran data ke pihak ketiga (seperti OpenAI atau Claude). Cukup unggah dokumen PDF Anda, dan AI akan siap menjawab pertanyaan Anda secara instan lengkap dengan kutipan halaman aslinya!

---

## 🌟 Mengapa ArsipPintar? (Project Overview)

Banyak instansi dan perusahaan menghadapi dilema besar: mereka ingin memanfaatkan kecerdasan AI untuk mencari informasi di ribuan dokumen internal mereka dengan cepat. Namun, aturan ketat mengenai **privasi data** melarang mereka mengunggah dokumen sensitif tersebut ke internet atau API eksternal. 

**ArsipPintar hadir sebagai solusi:** 
Dengan menggabungkan framework **LangChain (LCEL)** untuk pemrosesan dokumen dan **ChromaDB** sebagai database vektor lokal, sistem ini mampu mengindeks data baru secara bertahap tanpa tumpang tindih. Agar dapat berjalan mulus di komputer standar tanpa kartu grafis (GPU) mahal, proyek ini dioptimalkan menggunakan *Small Language Model* (SLM) ultra-ringan **Qwen2 (0.5B)** via **Ollama**. Semua kehebatan ini dibungkus dalam antarmuka web **Streamlit** yang interaktif dan mudah digunakan oleh siapa saja.

Hasilnya? Aplikasi tingkat produksi (*production-grade*) yang menjamin **0% kebocoran data**, memangkas biaya infrastruktur AI menjadi **nol rupiah**, serta menyajikan jawaban lintas dokumen yang instan dan akurat langsung dari laptop Anda.

---

## 🛠️ Tech Stack yang Digunakan

*   **Python** - Bahasa pemrograman utama sistem.
*   **LangChain (LCEL)** - Framework untuk mengatur alur data dan logika RAG.
*   **Ollama & Qwen2 (0.5B)** - Mesin AI lokal (Small Language Model) yang super ringan dan hemat CPU.
*   **ChromaDB** - Database vektor lokal untuk menyimpan dan mencari kecocokan teks dokumen.
*   **Streamlit** - Framework untuk membuat antarmuka (UI) web yang cantik dan interaktif dengan cepat.

---

## 🚀 Panduan Instalasi (Mudah untuk Pemula)

Ikuti langkah-masing di bawah ini untuk menjalankan ArsipPintar di komputer Anda sendiri:

### 1. Persiapan Awal
Pastikan Anda sudah menginstal **Python (versi 3.10 atau di atasnya)** dan **Ollama** di komputer Anda. jika belum punya Ollama, unduh gratis di [ollama.com](https://ollama.com).

### 2. Download Model AI Lokal
Buka Command Prompt (CMD) atau Terminal Anda, lalu jalankan perintah berikut untuk mengunduh otak AI yang akan kita gunakan:
```bash
ollama run qwen2:0.5b
```
Tunggu proses download selesai (ukurannya sangat kecil, hanya sekitar 350 MB). Jika sudah selesai, Anda bisa menutup CMD tersebut.

3. Clone Repositori Ini
Buka terminal baru di folder komputer tempat Anda ingin menyimpan proyek ini, lalu ketik:
```bash
git clone [https://github.com/username-anda/local-rag-arsippintar.git](https://github.com/username-anda/local-rag-arsippintar.git)
cd local-rag-arsippintar
```

4. Buat Virtual Environment & Instal Library
Biar rapi dan tidak bentrok dengan proyek lain, buat ruang kerja virtual lalu instal semua kebutuhan library-nya:
```bash

# Membuat virtual environment
python -m venv rag_venv

# Mengaktifkan virtual environment (Windows)
rag_venv\Scripts\activate

# Mengaktifkan virtual environment (Mac/Linux)
source rag_venv/bin/activate

# Menginstal semua library pendukung
pip install -r requirements.txt
```

💻 Cara Menjalankan Aplikasi
Setelah semua langkah instalasi di atas selesai, Anda tinggal menyalakan aplikasi web ArsipPintar dengan perintah berikut di terminal yang masih aktif:
```bash
streamlit run app_web.py
```
Aplikasi akan otomatis terbuka di browser Anda (biasanya di alamat http://localhost:8501).


📝 Cara Penggunaan:
Pada menu di sebelah kiri, unggah satu atau beberapa file PDF contoh Anda (misal: SOP perusahaan atau jurnal ilmiah).

Klik tombol "Proses & Tanam ke Otak AI". Tunggu beberapa saat hingga muncul notifikasi sukses.

Sekarang, beralihlah ke kolom obrolan di sebelah kanan. Ketik pertanyaan apa saja terkait dokumen yang Anda unggah tadi.

ArsipPintar akan menjawab pertanyaan Anda, lengkap dengan kolom Expander di bawahnya yang menunjukkan dokumen mana dan halaman berapa yang dijadikan rujukan jawaban tersebut!

💡 Catatan Optimasi Hardware
Proyek ini sengaja dikonfigurasi menggunakan model Qwen2 (0.5B) dan ukuran potongan teks (chunk size) sebesar 300 token agar ramah terhadap komputer berspesifikasi standar (bahkan komputer yang hanya menggunakan komputasi berbasis CPU jadul tanpa GPU diskret). Jika Anda memiliki spesifikasi komputer yang lebih tinggi, Anda bisa mencoba meningkatkan modelnya ke versi yang lebih besar seperti phi3:mini atau llama3 melalui file app_web.py.

🤝 Kontribusi
Punya ide untuk membuat ArsipPintar jadi lebih pintar lagi? Jangan ragu untuk melakukan Fork repositori ini, membuat branch baru, dan mengirimkan Pull Request. Semua kontribusi dan saran perbaikan sangat diapresiasi!

Dibuat dengan 💻 dan ☕ oleh Asfriansah.

import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. KONFIGURASI HALAMAN WEB
st.set_page_config(page_title="Local Enterprise Knowledge Base", page_icon="🏢", layout="wide")

# Folder lokal untuk menyimpan berkas fisik PDF yang diunggah
FOLDER_DOKUMEN = "dokumen_input"
FOLDER_DB = "chroma_db"
os.makedirs(FOLDER_DOKUMEN, exist_ok=True)

# 2. INSIALISASI AI LOKAL (Fungsi di-cache agar web tidak reload lambat)
@st.cache_resource
def dapatkan_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory=FOLDER_DB, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Ambil 3 chunk paling relevan
    llm = ChatOllama(model="qwen2:0.5b", temperature=0)
    
    system_prompt = (
        "Anda adalah asisten AI korporat yang bertugas menyajikan informasi dokumen secara akurat.\n"
        "Gunakan potongan dokumen berikut untuk menjawab pertanyaan di akhir.\n"
        "Sebutkan juga nama dokumen sumber yang relevan jika ada di dalam konteks.\n"
        "Jika jawaban tidak ada di dalam dokumen, katakan sejujurnya bahwa Anda tidak tahu.\n\n"
        "Konteks Dokumen:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        # Menggabungkan konten teks dan menyisipkan metadata nama file untuk sitasi sumber
        format_teks = []
        for doc in docs:
            sumber = doc.metadata.get('source', 'Dokumen Tidak Diketahui').split(os.sep)[-1]
            format_teks.append(f"[Sumber: {sumber}]\n{doc.page_content}")
        return "\n\n".join(format_teks)
        
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# --- LAYOUT UTAMA STREAMLIT (Tampilan Kolom Kiri & Kanan) ---
st.title("🏢 Local Enterprise Knowledge Base")
st.write("Sistem Manajemen Pengetahuan Berbasis AI Lokal (100% Privacy-First & Offline)")
st.divider()

kolom_kiri, kolom_kanan = st.columns([1, 2])

# ==================== KOLOM KIRI: MANAGEMENT DOKUMEN ====================
with kolom_kiri:
    st.header("📁 Dokumen Manajer")
    
    # Komponen Upload File PDF
    files_diunggah = st.file_uploader(
        "Unggah Dokumen PDF Baru", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    tombol_proses = st.button("🚀 Proses & Tanam ke Otak AI", use_container_width=True)
    
    if tombol_proses and files_diunggah:
        with st.spinner("Sedang membaca teks dan menghitung matematika vektor..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            semua_potongan_teks = []
            
            for file_pdf in files_diunggah:
                # Simpan file fisik sementara ke folder lokal
                path_simpan = os.path.join(FOLDER_DOKUMEN, file_pdf.name)
                with open(path_simpan, "wb") as f:
                    f.write(file_pdf.getbuffer())
                    
                # Ekstrak Teks menggunakan PyPDFLoader
                loader = PyPDFLoader(path_simpan)
                dokumen = loader.load()
                potongan = text_splitter.split_documents(dokumen)
                semua_potongan_teks.extend(potongan)
            
            if semua_potongan_teks:
                # Tambahkan data ke ChromaDB (Metode Tambah / Append)
                vector_store = Chroma(persist_directory=FOLDER_DB, embedding_function=embeddings)
                vector_store.add_documents(semua_potongan_teks)
                st.success(f"Berhasil melatih AI dengan {len(files_diunggah)} dokumen baru!")
                st.cache_resource.clear() # Reset cache agar model membaca database terbaru
            else:
                st.error("Gagal mengekstrak teks. Pastikan PDF bukan gambar hasil scan.")

    # Menampilkan daftar file yang sudah terindeks saat ini
    st.subheader("Database Pengetahuan Saat Ini:")
    if os.path.exists(FOLDER_DOKUMEN):
        daftar_file = [f for f in os.listdir(FOLDER_DOKUMEN) if f.endswith('.pdf')]
        if daftar_file:
            for file in daftar_file:
                st.caption(f"✅ {file}")
        else:
            st.info("Belum ada dokumen yang tersimpan di dalam database.")
    else:
        st.info("Belum ada dokumen yang tersimpan di dalam database.")

# ==================== KOLOM KANAN: TANYA JAWAB AI ====================
with kolom_kanan:
    st.header("💬 Konsultasi Asisten AI")
    
    # Load RAG Chain utama
    try:
        chain, retriever = dapatkan_rag_chain()
        st.caption("🟢 Status: Koneksi Ollama & Database Vektor Aktif")
    except Exception as e:
        st.error("🔴 Status: Gagal terhubung ke Ollama. Pastikan aplikasi Ollama Anda menyala di background.")
        chain = None

    st.divider()
    
    # Input Pertanyaan
    pertanyaan = st.text_input(
        "Ajukan pertanyaan cerdas lintas dokumen Anda di sini:", 
        placeholder="Contoh: Apa kebijakan lembur terbaru menurut dokumen yang ada?"
    )
    
    if pertanyaan and chain:
        with st.spinner("AI sedang membaca dokumen referensi..."):
            try:
                # 1. Cari potongan dokumen asli untuk dilampirkan sebagai kutipan referensi asli
                dokumen_relevan = retriever.invoke(pertanyaan)
                
                # 2. Generasi jawaban utama dari Phi-3
                jawaban = chain.invoke(pertanyaan)
                
                # Tampilkan Jawaban Utama
                st.markdown("### 📝 Jawaban AI:")
                st.info(jawaban)
                
                # Tampilkan Sumber Referensi (Citations) asli dari file berkas demi transparansi
                st.markdown("#### 🔍 Dokumen Referensi yang Digunakan:")
                for i, doc in enumerate(dokumen_relevan):
                    nama_sumber = doc.metadata.get('source', 'Dokumen').split(os.sep)[-1]
                    hal = doc.metadata.get('page', 0) + 1
                    with st.expander(f"Kutipan {i+1}: {nama_sumber} (Halaman {hal})"):
                        st.write(f"_\"{doc.page_content}\"_\n")
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")
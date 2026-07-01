import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

def proses_dokumen():
    # 1. Tentukan file PDF yang akan dibaca
    nama_file = "sampel.pdf"
    
    if not os.path.exists(nama_file):
        print(f"Error: File '{nama_file}' tidak ditemukan di folder ini!")
        return

    print("1. Membaca file PDF...")
    loader = PyPDFLoader(nama_file)
    dokumen = loader.load()
    print(f"   Berhasil membaca {len(dokumen)} halaman.")

    # 2. Potong teks menjadi chunk kecil (500 karakter per chunk)
    print("2. Memotong teks menjadi bagian-bagian kecil (chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    potongan_teks = text_splitter.split_documents(dokumen)
    print(f"   Dihasilkan {len(potongan_teks)} potongan teks.")

    # 3. Inisialisasi model embedding dari Ollama yang sudah kita download
    print("3. Menghubungkan ke model embedding Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Ubah teks jadi vektor dan simpan ke folder 'chroma_db' lokal
    print("4. Mengubah teks menjadi vektor dan menyimpan ke ChromaDB lokal...")
    folder_db = "chroma_db"
    
    # Membuat database dan menyimpan data
    Chroma.from_documents(
        documents=potongan_teks, 
        embedding=embeddings, 
        persist_directory=folder_db
    )
    print(f"   Sukses! Database vektor berhasil disimpan di folder '{folder_db}'.")

if __name__ == "__main__":
    proses_dokumen()
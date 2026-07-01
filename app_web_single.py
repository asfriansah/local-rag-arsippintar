import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Fungsi untuk menggabungkan potongan teks dokumen
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Konfigurasi halaman Web Streamlit
st.set_page_config(page_title="Local Smart Reader", page_icon="🤖", layout="centered")

st.title("🤖 Local Smart Document Reader")
st.subheader("Privacy-First RAG Portfolio Project")
st.write("Tanyakan apa saja mengenai dokumen `sampel.pdf` yang telah diproses.")

st.divider()

# Fungsi untuk memuat komponen RAG (di-cache agar web berjalan cepat)
@st.cache_resource
def inisialisasi_rag():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="phi3:mini", temperature=0)
    
    system_prompt = (
        "Anda adalah asisten pintar penyaji informasi dokumen.\n"
        "Gunakan potongan dokumen berikut untuk menjawab pertanyaan di akhir.\n"
        "Jika Anda tidak tahu jawabannya atau tidak ada di dokumen, katakan saja bahwa Anda tidak tahu, "
        "jangan mengarang jawaban.\n\n"
        "Konteks Dokumen:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Panggil fungsi inisialisasi
try:
    with st.spinner("Sedang memuat model AI lokal... Harap tunggu..."):
        chain = inisialisasi_rag()
    st.success("Model AI Lokal & Database Vektor Siap!")
except Exception as e:
    st.error(f"Gagal memuat komponen AI. Pastikan aplikasi Ollama Anda sudah menyala. Error: {e}")

st.divider()

# Input pertanyaan dari user di halaman web
pertanyaan = st.text_input("Masukkan pertanyaan Anda seputar dokumen:", placeholder="Contoh: Apa kesimpulan dari dokumen ini?")

if pertanyaan:
    with st.spinner("AI sedang membaca dokumen dan merumuskan jawaban..."):
        try:
            # Jalankan pencarian RAG
            jawaban = chain.invoke(pertanyaan)
            
            # Tampilkan Jawaban di Box yang rapi
            st.markdown("### 📝 Jawaban AI:")
            st.info(jawaban)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses jawaban: {e}")

st.caption("Aplikasi ini berjalan 100% lokal. Data Anda aman dan tidak dikirim ke internet.")
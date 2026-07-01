from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    # Fungsi pembantu untuk menggabungkan potongan teks dokumen
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print("1. Menginisialisasi komponen AI lokal (Versi LCEL)...")
    
    # 1. Panggil database vektor ChromaDB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embeddings
    )
    
    # 2. Jadikan database sebagai retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. Panggil SLM Phi-3 via ChatOllama
    llm = ChatOllama(model="phi3:mini", temperature=0)

    # 4. Buat template instruksi (Prompt)
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

    # 5. Bangun RAG Chain menggunakan LCEL (Modern & Bebas Modul Error)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- SISTEM SIAP! ---")
    print("Ketik 'keluar' untuk menyudahi sesi tanya jawab.\n")

    # 6. Loop Obrolan
    while True:
        pertanyaan = input("Pertanyaan Anda: ")
        if pertanyaan.lower() == 'keluar':
            print("Terima kasih telah menggunakan Local RAG!")
            break
            
        if pertanyaan.strip() == "":
            continue

        print("\nAI sedang membaca dokumen dan memikirkan jawaban...")
        
        # Jalankan rantai perintah
        jawaban = rag_chain.invoke(pertanyaan)
        
        print("\nJawaban AI:")
        print(jawaban)
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
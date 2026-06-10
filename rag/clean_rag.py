# clean_rag.py
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

VECTOR_STORE_PATH = "data/vector_store"

def deduplicate_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ Nessun vector store trovato nel percorso specificato.")
        return

    print("📦 Caricamento del Vector Store in corso...")
    # Usiamo OllamaEmbeddings solo come configurazione per il caricamento,
    # non faremo chiamate reali al server per l'istanza.
    embeddings = OllamaEmbeddings(model="qwen3") 
    
    db = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # Accediamo al dizionario interno dei documenti {id_documento: Oggetto_Document}
    docstore = db.docstore._dict
    print(f"📊 Documenti totali attuali nel database: {len(docstore)}")

    seen_chunks = set()
    ids_to_delete = []

    # Analizziamo i documenti e tracciamo gli ID di quelli duplicati
    for doc_id, doc in docstore.items():
        url = doc.metadata.get('url', '')
        content_hash = hash(doc.page_content.strip())
        # Creiamo una chiave univoca basata su URL + contenuto del frammento
        unique_key = f"{url}_{content_hash}"

        if unique_key in seen_chunks:
            ids_to_delete.append(doc_id)
        else:
            seen_chunks.add(unique_key)

    print(f"🧹 Frammenti duplicati individuati: {len(ids_to_delete)}")

    # Se ci sono duplicati, li eliminiamo direttamente per ID
    if ids_to_delete:
        print(f"🗑️ Eliminazione di {len(ids_to_delete)} ID dall'indice FAISS...")
        db.delete(ids_to_delete)
        
        print("💾 Salvataggio del Vector Store pulito...")
        db.save_local(VECTOR_STORE_PATH)
        print(f"✅ Pulizia completata! Rimasti {len(db.docstore._dict)} documenti unici.")
    else:
        print("✨ Il tuo Vector Store era già perfettamente pulito. Nessuna modifica necessaria.")

if __name__ == "__main__":
    deduplicate_vector_store()
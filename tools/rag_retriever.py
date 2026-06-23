# tools/rag_retriever.py
import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
import tempfile
import numpy as np

# Usa l'import corretto per la nuova versione
try:
    from langchain_ollama import OllamaEmbeddings
    print("✅ Using langchain_ollama")
except ImportError:
    # Fallback per versione vecchia
    from langchain_community.embeddings import OllamaEmbeddings
    print("⚠️ Using langchain_community (deprecated)")

class RAGRetriever:
    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = persist_dir
        
        # USA nomic-embed-text per embeddings (supporta embeddings!)
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # <-- Modello per embeddings
            base_url="http://localhost:11434"
        )
        
        self.vectorstore = None
        self._load_or_create()
    
    def _load_or_create(self):
        """Carica vector store esistente o ne crea uno nuovo"""
        if os.path.exists(self.persist_dir) and os.path.exists(os.path.join(self.persist_dir, "index.faiss")):
            try:
                from langchain_community.vectorstores import FAISS
                self.vectorstore = FAISS.load_local(
                    self.persist_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded existing vector store from {self.persist_dir}")
                return
            except Exception as e:
                print(f"⚠️ Could not load vector store: {e}")
        
        # Crea nuovo vector store
        os.makedirs(self.persist_dir, exist_ok=True)
        from langchain_community.vectorstores import FAISS
        
        # Inizializza con embedding di test
        test_embedding = self.embeddings.embed_query("Initialization")
        print(f"✅ Embeddings working! Dimension: {len(test_embedding)}")
        
        # Crea vector store vuoto
        self.vectorstore = FAISS.from_texts(
            ["RAG system initialized for blog posts."], 
            self.embeddings
        )
        self._save()
        print(f"✅ Created new vector store at {self.persist_dir}")
    
    @traceable(name="RAG-AddDocuments", run_type="chain", tags=["rag", "indexing"])
    def add_documents(self, documents: List[Dict]) -> int:
        """Aggiunge documenti al vector store"""
        if not documents:
            return 0
        
        # Recupera URL già presenti nel vector store per evitare duplicati
        existing_urls = set()
        if self.vectorstore and hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
            for existing_doc in self.vectorstore.docstore._dict.values():
                u = existing_doc.metadata.get('url')
                if u:
                    existing_urls.add(u)
        
        from langchain_community.vectorstores import FAISS
        
        texts = []
        metadatas = []
        
        for doc in documents:
            url = doc.get('url', '')
            if url and url in existing_urls:
                # Documento già indicizzato nel database, lo saltiamo
                continue
                
            content = doc.get('content', '')
            if not content or len(content) < 100:
                continue
            
            # Split in chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(content)
            
            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    texts.append(chunk)
                    metadatas.append({
                        'url': url,
                        'title': doc.get('title', '')[:100],
                        'source': doc.get('source', 'web_search')
                    })
        
        if texts:
            self.vectorstore.add_texts(texts, metadatas=metadatas)
            self._save()
            print(f"✅ Added {len(texts)} chunks from {len(documents)} documents")
        
        return len(texts)
    
    @traceable(name="RAG-Retrieve", run_type="retriever", tags=["rag", "retrieval"])
    def retrieve(self, query: str, k: int = 5, kg_context: str = None) -> List[Dict]:
        """Retrieves relevant documents, ensuring they are unique by URL/title"""
        if self.vectorstore is None:
            print("⚠️ Vectorstore not initialized")
            return []
        
        # Espandi query
        expanded_query = query
        if kg_context:
            expanded_query = f"{query}\n\nRelated: {kg_context[:500]}"
        
        try:
            # Recupera più chunk del necessario per poter fare deduplicazione per documento
            k_search = k * 4
            results = self.vectorstore.similarity_search_with_score(expanded_query, k=k_search)
            
            documents = []
            seen_urls = set()
            seen_titles = set()
            
            for doc, score in results:
                # Salta documento placeholder o di inizializzazione
                if "placeholder" in doc.page_content.lower() or "initialization" in doc.page_content.lower():
                    continue
                
                url = doc.metadata.get('url', '')
                title = doc.metadata.get('title', '')
                
                # Deduplica per URL (o per titolo se manca l'URL)
                if url:
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                elif title:
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                
                documents.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': float(1 - score)  # Normalizza score
                })
                
                # Raggiunto il limite di k documenti unici, ci fermiamo
                if len(documents) >= k:
                    break
            
            return documents
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return []
    
    def _save(self):
        """Salva vector store su disco"""
        try:
            from langchain_community.vectorstores import FAISS
            self.vectorstore.save_local(self.persist_dir)
        except Exception as e:
            print(f"⚠️ Could not save vector store: {e}")

# Istanza globale
_rag_instance = None

def get_rag_retriever():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGRetriever()
    return _rag_instance

@traceable(name="RAG-RetrieveWrapper", run_type="retriever", tags=["rag"])
def rag_retrieve(query: str, k: int = 5, kg_context: str = None) -> List[Dict]:
    """Tool function per RAG retrieval"""
    retriever = get_rag_retriever()
    return retriever.retrieve(query, k, kg_context)

@traceable(name="RAG-AddDocumentsWrapper", run_type="chain", tags=["rag"])
def rag_add_documents(documents: List[Dict]) -> int:
    """Tool function per aggiungere documenti al RAG"""
    retriever = get_rag_retriever()
    return retriever.add_documents(documents)
# tools/rag_retriever.py
import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    
    def add_documents(self, documents: List[Dict]) -> int:
        """Aggiunge documenti al vector store"""
        if not documents:
            return 0
        
        from langchain_community.vectorstores import FAISS
        
        texts = []
        metadatas = []
        
        for doc in documents:
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
                        'url': doc.get('url', ''),
                        'title': doc.get('title', '')[:100],
                        'source': doc.get('source', 'web_search')
                    })
        
        if texts:
            self.vectorstore.add_texts(texts, metadatas=metadatas)
            self._save()
            print(f"✅ Added {len(texts)} chunks from {len(documents)} documents")
        
        return len(texts)
    
    def retrieve(self, query: str, k: int = 5, kg_context: str = None) -> List[Dict]:
        """Retrieves relevant documents"""
        if self.vectorstore is None:
            print("⚠️ Vectorstore not initialized")
            return []
        
        # Espandi query
        expanded_query = query
        if kg_context:
            expanded_query = f"{query}\n\nRelated: {kg_context[:500]}"
        
        try:
            results = self.vectorstore.similarity_search_with_score(expanded_query, k=k)
            
            documents = []
            for doc, score in results:
                # Salta documento placeholder
                if "placeholder" in doc.page_content.lower() or "initialization" in doc.page_content.lower():
                    continue
                documents.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': float(1 - score)  # Normalizza score
                })
            
            print(f"📚 Retrieved {len(documents)} relevant documents")
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

def rag_retrieve(query: str, k: int = 5, kg_context: str = None) -> List[Dict]:
    """Tool function per RAG retrieval"""
    retriever = get_rag_retriever()
    return retriever.retrieve(query, k, kg_context)

def rag_add_documents(documents: List[Dict]) -> int:
    """Tool function per aggiungere documenti al RAG"""
    retriever = get_rag_retriever()
    return retriever.add_documents(documents)
# Reranker.py
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Inizializza il reranker con un modello CrossEncoder.
        Modelli consigliati:
        - "cross-encoder/ms-marco-MiniLM-L-6-v2" (veloce, buono)
        - "cross-encoder/ms-marco-electra-base" (più accurato)
        - "BAAI/bge-reranker-base" (multilingua)
        """
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    
    def rerank(self, query, documents, top_k=5):
        """
        Rerank i documenti in base alla rilevanza per la query.
        
        Args:
            query: stringa della query
            documents: lista di dict con campo 'text' o 'page_content'
            top_k: numero di documenti da restituire
        
        Returns:
            Lista di documenti riordinati con nuovi score
        """
        if not documents:
            return []
        
        # Prepara coppie (query, documento)
        pairs = []
        for doc in documents:
            text = doc.get("text", doc.get("page_content", ""))
            if not text:
                text = doc.get("content", "")
            pairs.append((query, text[:512]))  # Limita lunghezza
        
        # Calcola similarity scores
        scores = self.model.predict(pairs)
        
        # Combina scores con documenti
        scored_docs = []
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(scores[i])
            doc_copy["original_score"] = doc.get("score", 0)
            # Score combinato (peso 70% rerank, 30% original)
            doc_copy["combined_score"] = 0.7 * float(scores[i]) + 0.3 * doc_copy["original_score"]
            scored_docs.append(doc_copy)
        
        # Ordina per combined score
        scored_docs.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return scored_docs[:top_k]
    
    def batch_rerank(self, queries, documents_list, top_k=5):
        """Rerank multiple query-document pairs in batch"""
        results = []
        for query, docs in zip(queries, documents_list):
            results.append(self.rerank(query, docs, top_k))
        return results
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import ollama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.web_search import web_search
from tools.extract_clean_docs import _extract_clean_docs
from tools.summarize_results import _summarize_results


class ResearchAgent:
    def __init__(self, name, state):
        self.name = name
        self.state = state

    def _generate(self, prompt: str) -> str:
        """Generate text using the configured model"""
        model = self.state.get("model")
        model_name = self.state.get("model_name", "llama3.1")
        
        if hasattr(model, "chat"):
            response = model.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]

        if hasattr(model, "generate"):
            response = model.generate(prompt)
            return response.text if hasattr(response, "text") else str(response)

        if isinstance(model, str):
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]

        raise TypeError("model must be an Ollama Client, a model name string, or expose generate().")
    
    def _generate_doc_hash(self, text, url):
        """Genera un hash univoco per il documento"""
        import hashlib
        content = f"{url}:{text[:500]}"  # Primi 500 caratteri + URL
        return hashlib.sha256(content.encode()).hexdigest()

    def _rag_count(self, vector_store) -> int:
        collection = getattr(vector_store, "_collection", None)
        if collection and hasattr(collection, "count"):
            return collection.count()
        return 0

    def _get_vector_store(self) -> Chroma:
        rag_config = self.state.get("rag_config", {})
        persist_dir = rag_config.get("persist_dir", "data/chroma")
        collection_name = rag_config.get("collection", "news_docs")
        embedding_model = rag_config.get("embedding_model", "nomic-embed-text")

        embeddings = OllamaEmbeddings(model=embedding_model)
        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    def _is_recent(self, doc: Dict, max_age_hours: int = 72) -> bool:
        """Check if document is recent enough"""
        fetched_at = doc.get("fetched_at")
        if not fetched_at:
            return False
        try:
            dt = datetime.fromisoformat(fetched_at)
        except ValueError:
            return False
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
        return age_hours <= max_age_hours

    def perform_research(self, query: str, max_steps: int = 2, use_rag: bool = True) -> Dict:
        reasoning_trace = []
        tool_outputs = {}
        from tools.summarize_results import _summarize_results_with_extraction
        # Handle review feedback
        review_note = (self.state.get("content_feedback_detail") or "").strip()
        if review_note:
            reasoning_trace.append({
                "step": "review_note",
                "thought": f"Using review feedback: {review_note}"
            })
            # Incorporate review note into query if it adds context
            if len(review_note) < 100:  # Avoid overly long queries
                query = f"{query} {review_note}"

        # Clean and prepare query
        current_query = query.split("\n")[0].strip()[:400]

        rag_config = self.state.get("rag_config", {})
        max_age_hours = rag_config.get("max_age_hours", 72)
        use_rag = rag_config.get("use_rag", use_rag)

        vector_store = self._get_vector_store() if use_rag else None
        all_docs = []
        rag_results = []

        for step in range(max_steps):
            reasoning_trace.append({
                "step": step + 1,
                "thought": f"Searching for: {current_query}"
            })

            # Web search
            search_results = web_search.invoke({"query": current_query, "max_results": 15})
            tool_outputs["search"] = search_results
            summarized = _summarize_results_with_extraction(search_results, max_items=5)
            
            clean_docs = _extract_clean_docs(search_results, max_items=15)
            now_iso = datetime.now(timezone.utc).isoformat()
            for doc in clean_docs:
                doc["fetched_at"] = now_iso
                doc["source_type"] = "web"

            # RAG operations
            if use_rag and vector_store:
                # Store new documents
                self._rag_append(clean_docs, query, vector_store)
                # Retrieve relevant documents
                rag_results = self._rag_search(current_query, topic_filter=query, vector_store=vector_store)
                for doc in rag_results:
                    doc["source_type"] = "rag"
                
                # Combine results prioritizing recent RAG docs
                recent_rag = [d for d in rag_results if self._is_recent(d, max_age_hours)]
                if recent_rag:
                    all_docs = recent_rag[:15]
                else:
                    all_docs = clean_docs[:15]
            else:
                all_docs = clean_docs[:15]

            tool_outputs["clean_docs"] = all_docs
            tool_outputs["rag_results"] = rag_results
            if vector_store:
                tool_outputs["rag_doc_count"] = self._rag_count(vector_store)

            reasoning_trace.append({"observation": summarized})

            # Stop if we have enough info
            if len(summarized) >= 3:
                break

            # Refine query for next iteration
            current_query = f"{query} ultime notizie"

        # Generate final verified summary
        verified_info = self._generate_verified_info(query, all_docs[:10])

        return {
            "verified_info": verified_info,
            "reasoning_trace": reasoning_trace,
            "tool_outputs": tool_outputs
        }

    def _rag_append(self, clean_docs: List[Dict], query: str, vector_store: Optional[Chroma] = None):
        """Add documents to vector store"""
        if not clean_docs:
            return
            
        rag_config = self.state.get("rag_config", {})
        chunk_size = rag_config.get("chunk_size", 1000)
        chunk_overlap = rag_config.get("chunk_overlap", 100)

        if vector_store is None:
            vector_store = self._get_vector_store()

        now_iso = datetime.now(timezone.utc).isoformat()
        docs = []
        for doc in clean_docs:
            text = doc.get("text", "")
            url = doc.get("url")
            title = doc.get("title")
            if not text or not url:
                continue

            doc_hash = self._generate_doc_hash(text, url)

            existing = vector_store.similarity_search_with_score(
            f"hash:{doc_hash}", k=1, filter={"hash": doc_hash}
        )
            if existing and existing[0][1] > 0.95:
                print(f"[RAG] Skipping duplicate document: {url}")
                continue
            
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "title": title,
                        "fetched_at": now_iso,
                        "topic": query,
                        "hash": doc_hash,
                        "content_length": len(text)
                    },
                )
            )

        if docs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            splits = splitter.split_documents(docs)
            vector_store.add_documents(splits)
            print(f"[RAG] Added {len(splits)} chunks for topic '{query}'")


    def _generate_verified_info(self, query: str, docs: List[Dict]) -> str:
        """Generate verified summary from documents"""
        if not docs:
            return f"Verified info about {query} based on 0 sources."

        context = "\n\n".join([
            f"Fonte {i+1}: {doc.get('title', 'Senza titolo')}\n{doc.get('text', '')[:500]}"
            for i, doc in enumerate(docs[:5]) if doc.get('text')
        ])

        prompt = f"""
        Basandoti ESCLUSIVAMENTE sulle seguenti fonti, fornisci un riassunto verificato di 3-5 frasi in italiano su: {query}

        FONTI:
        {context}

        RIASSUNTO VERIFICATO (solo fatti presenti nelle fonti, in italiano):
        """

        return self._generate(prompt)

    def debug_rag_store(self) -> int:
        """Debug method to inspect vector store contents"""
        vector_store = self._get_vector_store()
        count = self._rag_count(vector_store)
        print(f"[RAG DEBUG] Total documents in store: {count}")

        if count > 0:
            test_results = vector_store.similarity_search_with_score("test", k=1)
            if test_results:
                from random import randrange
                print(f"[RAG DEBUG] Sample document metadata: {test_results[randrange(len(test_results))][randrange(len(test_results))].metadata}")
            else:
                print("[RAG DEBUG] Store has documents but search returns nothing")
        else:
            print("[RAG DEBUG] Store is empty!")

        return count
    
    def _generate_multi_queries(self, original_query, num_queries=3):
        """Genera query alternative per lo stesso argomento"""
        
        prompt = f"""
        Data la query originale: "{original_query}"
        Genera {num_queries} query di ricerca alternative ma correlate che potrebbero trovare 
        informazioni complementari o diverse prospettive su questo argomento.
        
        Restituisci SOLO le query, una per riga, senza numeri o spiegazioni.
        """
        
        response = self._generate(prompt)
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        
        # Aggiungi la query originale
        all_queries = [original_query] + queries[:num_queries]
        
        print(f"[MultiQuery] Generate {len(all_queries)} queries:")
        for q in all_queries:
            print(f"  - {q}")
        
        return all_queries

    def _multi_query_retrieval(self, query, vector_store, top_k=5):
        """Esegue retrieval con multiple query e combina i risultati"""
        
        # Genera query alternative
        queries = self._generate_multi_queries(query, num_queries=3)
        
        all_results = []
        seen_urls = set()
        
        for q in queries:
            results = vector_store.similarity_search_with_score(q, k=top_k)
            
            for doc, score in results:
                url = doc.metadata.get("source", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                all_results.append({
                    "text": doc.page_content,
                    "url": url,
                    "title": doc.metadata.get("title", ""),
                    "fetched_at": doc.metadata.get("fetched_at", ""),
                    "topic": doc.metadata.get("topic", ""),
                    "score": score,
                    "query_used": q
                })
        
        # Aggrega e deduplica
        return self._aggregate_multi_query_results(all_results, top_k)

    def _aggregate_multi_query_results(self, results, top_k):
        """Aggrega risultati da multiple query usando max score"""
        
        # Raggruppa per URL
        grouped = {}
        for r in results:
            url = r["url"]
            if url not in grouped:
                grouped[url] = r
            else:
                # Prendi lo score più alto tra le query
                grouped[url]["score"] = max(grouped[url]["score"], r["score"])
        
        # Converti in lista e ordina
        aggregated = list(grouped.values())
        aggregated.sort(key=lambda x: x["score"], reverse=True)
        
        return aggregated[:top_k]
    
    # research_agent.py - Aggiungi recency scoring

    def _calculate_recency_score(self, timestamp, max_age_days=30):
        """
        Calcola un recency score normalizzato tra 0 e 1.
        Più recente = punteggio più alto.
        """
        if not timestamp:
            return 0.5  # Valore neutro se non disponibile
        
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
                
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
                
            now = datetime.now(timezone.utc)
            age_hours = (now - dt).total_seconds() / 3600.0
            age_days = age_hours / 24.0
            
            # Decadimento esponenziale: score = e^(-age_days / half_life)
            half_life_days = 7  # Emivita di 7 giorni
            import numpy as np
            recency_score = np.exp(-age_days / half_life_days)
            
            # Normalizza tra 0 e 1
            recency_score = max(0, min(1, recency_score))
            
            return recency_score
            
        except Exception as e:
            print(f"[Recency] Error calculating recency: {e}")
            return 0.5

    def _score_document(self, doc, query, topic_filter=None):
        """
        Calcola score composito per un documento usando:
        - Similarità semantica (score originale)
        - Recency score
        - Topic relevance
        """
        final_score = doc.get("score", 0.5)
        
        # Recency component (peso 20%)
        timestamp = doc.get("fetched_at")
        if timestamp:
            recency = self._calculate_recency_score(timestamp)
            final_score = final_score * 0.8 + recency * 0.2
        
        # Topic relevance (peso 10% se topic_filter presente)
        if topic_filter:
            doc_topic = doc.get("topic", "")
            if doc_topic:
                from difflib import SequenceMatcher
                topic_similarity = SequenceMatcher(None, 
                    topic_filter.lower(), doc_topic.lower()).ratio()
                final_score = final_score * 0.9 + topic_similarity * 0.1
        
        return final_score

    # Aggiorna RAG_search per usare recency score
    def _rag_search(self, query, topic_filter=None, vector_store=None, use_reranker=True):
        """Versione migliorata con recency score e reranker opzionale"""
        
        rag_config = self.state.get("rag_config", {})
        top_k = rag_config.get("top_k", 15)
        score_threshold = rag_config.get("score_threshold", 0.35)
        
        if vector_store is None:
            vector_store = self._get_vector_store()
        
        # Multi-query retrieval
        results = self._multi_query_retrieval(query, vector_store, top_k=top_k * 2)
        
        # Applica recency scoring
        for doc in results:
            doc["raw_score"] = doc["score"]
            doc["score"] = self._score_document(doc, query, topic_filter)
            doc["recency_score"] = self._calculate_recency_score(doc.get("fetched_at"))
        
        # Filtra per threshold
        filtered = [d for d in results if d["score"] >= score_threshold]
        
        # Reranking con CrossEncoder se disponibile
        if use_reranker and len(filtered) > 0:
            try:
                from tools.Reranker import Reranker
                reranker = Reranker()
                filtered = reranker.rerank(query, filtered, top_k=top_k)
            except ImportError:
                print("[RAG] Reranker not available, using base scoring")
                filtered.sort(key=lambda x: x["score"], reverse=True)
                filtered = filtered[:top_k]
        else:
            filtered.sort(key=lambda x: x["score"], reverse=True)
            filtered = filtered[:top_k]
        
        # Aggiorna stato
        tool_outputs = self.state.setdefault("tool_outputs", {})
        tool_outputs["rag_results"] = filtered
        tool_outputs["rag_doc_count"] = self._rag_count(vector_store)
        
        return filtered
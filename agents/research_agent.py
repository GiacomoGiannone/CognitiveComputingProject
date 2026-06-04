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
            summarized = _summarize_results(search_results)
            
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
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "title": title,
                        "fetched_at": now_iso,
                        "topic": query,
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

    def _rag_search(self, query: str, topic_filter: Optional[str] = None, vector_store: Optional[Chroma] = None) -> List[Dict]:
            """Search vector store with soft topic filtering"""
            rag_config = self.state.get("rag_config", {})
            top_k = rag_config.get("top_k", 15)
            score_threshold = rag_config.get("score_threshold", 0.35)

            if vector_store is None:
                vector_store = self._get_vector_store()

            # Get more results for soft filtering
            results = vector_store.similarity_search_with_score(
                query,
                k=top_k * 2,
            )

            filtered = []
            for d, score in results:
                if score < score_threshold:
                    continue
                    
                doc_topic = d.metadata.get("topic", "")
                final_score = score
                
                # Apply soft topic filter
                if topic_filter and doc_topic:
                    if topic_filter.lower() in doc_topic.lower():
                        final_score = score * 1.1
                    elif not any(word in doc_topic.lower() for word in query.lower().split()[:3]):
                        final_score = score * 0.7
                        
                filtered.append({
                    "text": d.page_content,
                    "url": d.metadata.get("source", "URL non disponibile"),  # Assicurati che l'URL sia presente
                    "title": d.metadata.get("title", "Senza titolo"),
                    "fetched_at": d.metadata.get("fetched_at"),
                    "score": final_score,
                    "topic": doc_topic,
                    "source_type": "rag"
                })

            filtered.sort(key=lambda x: x["score"], reverse=True)
            return filtered[:top_k]

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
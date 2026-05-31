from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.web_search import web_search
from tools.extract_clean_docs import _extract_clean_docs
from tools.summarize_results import _summarize_results

#import embeddings per RAG (da implementare)


class ResearchAgent:
    def __init__(self, name, state):
        self.name = name
        self.state = state  # Stato dell'agente, da implementare

    def perform_research(self, query, max_steps=2):
        # ReAct-style loop: Thought -> Action -> Observation
        # Nota sul pattern ReAct del progetto: Invece di creare un loop a livello di LangGraph
        # (che renderebbe la struttura dei nodi molto complessa all'inizio), il ciclo ReAct 
        # (Pensiero -> Azione -> Osservazione) è implementato come un loop "interno" a questo agente Python
        reasoning_trace = []
        tool_outputs = {}
        
        current_query = query
        if isinstance(current_query, str):
            current_query = current_query.split("\n")[0].strip()
            
            if len(current_query) > 400:
                current_query = current_query[:400]

        # Prima proviamo a fare retrieval dal vector store
        rag_results = self.RAG_search(current_query)
        if rag_results:
            tool_outputs["rag_results"] = rag_results
            tool_outputs["clean_docs"] = rag_results

        for step in range(max_steps):
            if rag_results:
                break
            reasoning_trace.append({
                "step": step + 1,
                "thought": f"Need recent sources for: {current_query}"
            })

            reasoning_trace.append({
                "action": "web_search",
                "action_input": {"query": current_query, "max_results": 15}
            })
            search_results = web_search.invoke({"query": current_query, "max_results": 15})
            tool_outputs["search"] = search_results

            summarized = _summarize_results(search_results)
            clean_docs = _extract_clean_docs(search_results, max_items=15)
            tool_outputs["clean_docs"] = clean_docs
            reasoning_trace.append({
                "observation": summarized
            })
            clean_docs_preview = []
            for doc in clean_docs:
                text = doc.get("text", "")
                clean_docs_preview.append({
                    "url": doc.get("url"),
                    "title": doc.get("title"),
                    "preview": text[:300]
                })
            reasoning_trace.append({
                "observation_clean_docs": clean_docs_preview
            })

            # Aggiorniamo il vector store con i nuovi clean_docs
            self.RAG_append(clean_docs, current_query)

            # Ora possiamo rifare la retrieval dal vector store
            rag_results = self.RAG_search(current_query)
            if rag_results:
                tool_outputs["rag_results"] = rag_results
                tool_outputs["clean_docs"] = rag_results

            if len(summarized) >= 3:
                break

            current_query = f"{query} ultime notizie"
            
        verified_info = f"Verified info about {query} based on {len(tool_outputs.get('clean_docs', []))} sources."
        return {
            "verified_info": verified_info,
            "reasoning_trace": reasoning_trace,
            "tool_outputs": tool_outputs
        }


    def update_state(self, new_state):
        # Logica per aggiornare lo stato dell'agente
        self.state = new_state

    #Vogliamo dare a questo agente la possibilita' di usare uno storage vettoriale da usare per il RAG.
    def RAG_append(self, clean_docs, query):
        rag_config = self.state.get("rag_config", {})
        chunk_size = rag_config.get("chunk_size", 1000)
        chunk_overlap = rag_config.get("chunk_overlap", 100)
        persist_dir = rag_config.get("persist_dir", "data/chroma")
        collection_name = rag_config.get("collection", "news_docs")
        embedding_model = rag_config.get("embedding_model", "nomic-embed-text")

        embeddings = OllamaEmbeddings(model=embedding_model)
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

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

    def RAG_search(self, query):
        rag_config = self.state.get("rag_config", {})
        top_k = rag_config.get("top_k", 5)
        score_threshold = rag_config.get("score_threshold", 0.35)
        persist_dir = rag_config.get("persist_dir", "data/chroma")
        collection_name = rag_config.get("collection", "news_docs")
        embedding_model = rag_config.get("embedding_model", "nomic-embed-text")

        embeddings = OllamaEmbeddings(model=embedding_model)
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

        results = vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter={"topic": query},
        )

        filtered = []
        for d, score in results:
            if score < score_threshold:
                continue
            filtered.append(
                {
                    "text": d.page_content,
                    "url": d.metadata.get("source"),
                    "title": d.metadata.get("title"),
                    "fetched_at": d.metadata.get("fetched_at"),
                    "score": score,
                }
            )

        self.state.setdefault("tool_outputs", {})["rag_results"] = filtered
        return filtered
from agents.research_agent import ResearchAgent
from tools.web_search import web_search
from tools.extract_clean_docs import _extract_clean_docs

# questo nodo aiuta il research agent quando riceve feedback negativo
# senza modificare la query principale: aggiunge nuove fonti al RAG

def rag_helper_node(state):
    print("[rag_helper] start")
    
    feedback = (state.get("content_feedback_detail") or "").strip()
    if not feedback:
        print("[rag_helper] no feedback, skipping")
        return state
    
    query = feedback
    topic = state.get("chosen_topic") or query
    
    # Fai una ricerca MIRATA sul feedback
    search_results = web_search.invoke({"query": query, "max_results": 15})
    clean_docs = _extract_clean_docs(search_results, max_items=15)
    
    # Aggiungi anche una ricerca sul topic originale
    topic_search = web_search.invoke({"query": topic, "max_results": 10})
    topic_docs = _extract_clean_docs(topic_search, max_items=10)
    clean_docs.extend(topic_docs)
    
    agent = ResearchAgent(name="rag_helper", state=state)
    agent.RAG_append(clean_docs, topic)
    
    # IMPORTANTE: resetta il flag per forzare nuova ricerca
    state["content_feedback"] = "no"  # Non approvato ma aggiunte fonti
    state["rag_helper_executed"] = True
    
    tool_outputs = state.setdefault("tool_outputs", {})
    tool_outputs["rag_helper_docs"] = clean_docs
    
    return state
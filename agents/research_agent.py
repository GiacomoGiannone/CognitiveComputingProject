from tools.web_search import web_search
from tools.extract_clean_docs import _extract_clean_docs
from tools.summarize_results import _summarize_results

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

        for step in range(max_steps):
            reasoning_trace.append({
                "step": step + 1,
                "thought": f"Need recent sources for: {current_query}"
            })

            reasoning_trace.append({
                "action": "web_search",
                "action_input": {"query": current_query, "max_results": 8}
            })
            search_results = web_search.invoke({"query": current_query, "max_results": 8})
            tool_outputs["search"] = search_results

            summarized = _summarize_results(search_results)
            clean_docs = _extract_clean_docs(search_results)
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

            if len(summarized) >= 3:
                break

            current_query = f"{query} ultime notizie"
            
        print("\n=== DEBUG: RISULTATI DELLA RICERCA WEB (Tavily) ===")
        print(f"Query usata: {current_query}")
        
        search_data = tool_outputs.get('search', {})
        results_list = search_data.get('results', []) if isinstance(search_data, dict) else search_data
        
        print(f"Numero di Fonti Trovate (Tavily): {len(results_list)}")
        for idx, doc in enumerate(results_list):
             if isinstance(doc, dict):
                 print(f"Fonte [{idx+1}]: {doc.get('title', 'N/A')}")
                 print(f"URL: {doc.get('url', 'N/A')}")
                 print(f"Tavily Snippet: {doc.get('content', 'N/A')}")
                 print("-" * 50)
                 
        print("\n=== DEBUG: TESTO COMPLETO ESTRATTO DALLE PAGINE (Trafilatura) ===")
        clean_docs = tool_outputs.get('clean_docs', [])
        print(f"Pagine di cui abbiamo scaricato il testo intero: {len(clean_docs)}")
        for idx, doc in enumerate(clean_docs):
            print(f"Articolo [{idx+1}]: {doc.get('title', 'N/A')}")
            print(f"URL: {doc.get('url', 'N/A')}")
            
            # Prendiamo il testo intero scaricato e ne stampiamo un'anteprima più lunga (es. i primi 600 caratteri)
            full_text = doc.get('text', 'N/A')
            preview = full_text[:600].replace('\n', ' ')
            print(f"Testo Estratto (anteprima): {preview}...\n[Continua... Totale caratteri: {len(full_text)}]")
            print("-" * 50)
             
        verified_info = f"Verified info about {query} based on {len(_summarize_results(tool_outputs.get('search', {})))} sources."
        return {
            "verified_info": verified_info,
            "reasoning_trace": reasoning_trace,
            "tool_outputs": tool_outputs
        }


    def update_state(self, new_state):
        # Logica per aggiornare lo stato dell'agente
        self.state = new_state
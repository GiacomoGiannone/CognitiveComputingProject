from langchain.tools import tool
from tavily import TavilyClient
import os
import trafilatura

@tool
def web_search(query, max_results = 5):
    """Search the web for recent sports news and return results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY non è impostata nelle variabili d'ambiente.")
    client = TavilyClient(api_key=api_key)
    
    # Rimuoviamo topic="news" perché l'algoritmo di news di Tavily 
    # tende a concentrarsi sulle fonti inglesi mainstream o deviare rispetto 
    # alle query specifiche in lingua italiana.
    # Usando il search classico avanzato, cercherà esplicitamente le nostre keyword italiane.
    results = client.search(
        query, 
        search_depth="advanced",
        max_results=max_results
    )
    
    if isinstance(results, dict) and "results" in results:
        for r in results["results"]:
            url = r.get("url")
            if url:
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        extracted = trafilatura.extract(downloaded)
                        if extracted and extracted.strip():
                            r["content"] = extracted
                except Exception:
                    # In caso di errore di rete o altro errore, manteniamo lo snippet originale
                    pass
                    
    return results
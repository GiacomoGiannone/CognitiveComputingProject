from langchain.tools import tool
from tavily import TavilyClient
import os

@tool
def web_search(query, max_results = 10):
    """Search the web for recent sports news and return results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY non è impostata nelle variabili d'ambiente.")
    client = TavilyClient(api_key=api_key)
    
    # Usiamo topic="news" per dire a Tavily di cercare solo notizie recenti,
    # ed evitiamo articoli obsoleti. Possiamo anche impostare search_depth="advanced" o i giorni (days=7).
    results = client.search(
        query, 
        max_results=max_results, 
        topic="news",
        days=15 # Cerca solo negli ultimi 15 giorni
    )
    return results
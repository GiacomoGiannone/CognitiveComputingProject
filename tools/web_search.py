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
    
    # Rimuoviamo topic="news" perché l'algoritmo di news di Tavily 
    # tende a concentrarsi sulle fonti inglesi mainstream o deviare rispetto 
    # alle query specifiche in lingua italiana.
    # Usando il search classico avanzato, cercherà esplicitamente le nostre keyword italiane.
    results = client.search(
        query, 
        search_depth="advanced",
        max_results=max_results
    )
    return results
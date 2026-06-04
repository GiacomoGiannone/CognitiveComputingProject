# summarize_results.py - Versione migliorata
from tools.extract_clean_docs import _extract_clean_docs

def _summarize_results_with_extraction(results, max_items=5):
    """
    Estrae prima i documenti puliti, poi li riassume.
    Output più pulito e basato sul contenuto effettivo.
    """
    if not isinstance(results, dict):
        return []
    
    # Prima estrai i documenti puliti
    clean_docs = _extract_clean_docs(results, max_items=max_items)
    
    if not clean_docs:
        return []
    
    # Poi crea un summary basato sul contenuto estratto
    summary = []
    for doc in clean_docs:
        text = doc.get("text", "")
        # Prendi prime 300 caratteri come snippet significativo
        snippet = text[:300].strip()
        if len(text) > 300:
            snippet += "..."
        
        summary.append({
            "title": doc.get("title"),
            "url": doc.get("url"),
            "snippet": snippet,
            "content_length": len(text)
        })
    
    return summary

# Versione originale per backward compatibility
def _summarize_results(results, max_items=5):
    """Versione originale per compatibilità"""
    if isinstance(results, dict):
        items = results.get("results", [])
        summary = []
        for item in items[:max_items]:
            summary.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content")
            })
        return summary
    return []
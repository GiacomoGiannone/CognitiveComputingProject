def _summarize_results(results, max_items=5):
     if isinstance(results,dict):
        items = results.get("results", [])
        summary = []
        for item in items[:max_items]:
            summary.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content")
            })
        return summary
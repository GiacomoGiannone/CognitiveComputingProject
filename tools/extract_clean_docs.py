import trafilatura

def _extract_clean_docs(results, max_items=3):
    items = results.get("results", []) if isinstance(results, dict) else []
    clean_docs = []
    for item in items[:max_items]:
        url = item.get("url")
        if not url:
            continue
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            continue
        extracted = trafilatura.extract(downloaded)
        if not extracted:
            continue
        clean_docs.append({
            "url": url,
            "title": item.get("title"),
            "text": extracted
        })
    return clean_docs

# agents/writer_agent.py
import ollama
import re

def extract_relevant_content(content: str, topic: str, max_chars: int = 2500) -> str:
    """
    Estrae le parti più rilevanti del contenuto in base al topic.
    Se il contenuto è breve (< 3000 caratteri), lo restituisce tutto.
    Altrimenti, lo divide in paragrafi e seleziona quelli con più parole in comune con il topic,
    fino a raggiungere il limite max_chars.
    """
    if not content:
        return ""
    if len(content) <= 3000:
        return content

    # Parole chiave del topic pulite (ignora stop words comuni)
    stop_words = {
        'il', 'lo', 'la', 'i', 'gli', 'le', 'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 
        'and', 'the', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'an', 'a', 'is', 'how', 
        'tips', 'improving', 'your', 'for', 'about'
    }
    topic_words = [w.lower() for w in re.findall(r'\b\w{3,}\b', topic) if w.lower() not in stop_words]
    if not topic_words:
        topic_words = topic.lower().split()

    # Dividi in paragrafi
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    if not paragraphs:
        return content[:max_chars]

    # Assegna un punteggio a ciascun paragrafo
    scored_paragraphs = []
    for idx, p in enumerate(paragraphs):
        p_lower = p.lower()
        score = 0
        for word in topic_words:
            # Parola intera vale di più
            score += len(re.findall(r'\b' + re.escape(word) + r'\b', p_lower)) * 2
            # Corrispondenza parziale vale meno
            if word in p_lower and not re.search(r'\b' + re.escape(word) + r'\b', p_lower):
                score += 1
        
        # Piccolo bonus per i paragrafi iniziali per mantenere l'introduzione se rilevante
        position_bonus = max(0, 5 - idx) * 0.1
        scored_paragraphs.append((score + position_bonus, idx, p))

    # Ordina per punteggio decrescente
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

    # Seleziona i paragrafi migliori fino al limite di caratteri
    selected_with_indices = []
    current_length = 0

    for score, idx, p in scored_paragraphs:
        if score <= 0.5 and current_length > 1000:
            continue
        if current_length + len(p) > max_chars:
            if not selected_with_indices:
                p = p[:max_chars]
                selected_with_indices.append((idx, p))
                break
            continue
        selected_with_indices.append((idx, p))
        current_length += len(p) + 2

    if not selected_with_indices:
        return content[:max_chars]

    # Riordina per mantenere la sequenzialità originale di lettura
    selected_with_indices.sort(key=lambda x: x[0])
    
    return "\n\n".join([p for _, p in selected_with_indices])


def writer_agent(state):
    """Writer agent - scrive il post basato sulla ricerca"""
    
    print("\n" + "="*50)
    print("✍️ WRITER AGENT STARTING")
    print("="*50)
    
    topic = state.get('current_topic')
    research = state.get('research_results', {})
    
    print(f"📝 Topic: {topic}")
    print(f"📚 Sources available: {research.get('num_sources', 0)}")
    
    # Se c'è un errore nella ricerca
    if research.get('error'):
        error_post = f"""
# Unable to Generate Post: {research.get('topic', 'No Topic')}

## Error
{research.get('research_summary', 'Unknown error')}

## What to do
1. Please try a different topic
2. Check that your Tavily API key is valid
3. Make sure you have internet connection

## Suggested topics instead:
- Basic sailing techniques for beginners
- Essential safety equipment for small boats
- How to choose your first sailboat
"""
        return {
            "draft_post": {
                "title": f"ERROR: Cannot research '{topic}'",
                "content": error_post,
                "topic": topic,
                "sources": [],
                "error": True
            }
        }
    
    # Se non ci sono fonti
    if research.get('num_sources', 0) == 0:
        no_sources_post = f"""
# Unable to Find Reliable Sources for: {topic}

## Problem
My research couldn't find any reliable sources for this specific topic.

## Research Summary
{research.get('research_summary', 'No research available')}

## Suggestions for next steps:
1. Try a broader topic (e.g., "sailing maintenance" instead of a specific winch model)
2. Check if your Tavily API key is working
3. Try again with a different topic

## Alternative topics you might like:
- Basic sailboat maintenance tips
- Essential knots every sailor should know
- Safety checks before leaving the dock

---
*This post was not generated due to lack of source material.*
"""
        return {
            "draft_post": {
                "title": f"No sources found: {topic}",
                "content": no_sources_post,
                "topic": topic,
                "sources": [],
                "error": True
            }
        }
    
    # Costruzione del prompt per il writer
    sources_text = ""
    for i, doc in enumerate(research.get('sources', [])[:3]):
        content = doc.get('content', '')
        relevant_content = extract_relevant_content(content, topic, max_chars=2500)
        sources_text += f"""
SOURCE {i+1}:
Title: {doc.get('title', 'No title')}
URL: {doc.get('url', 'No URL')}
Content:
{relevant_content}
---
"""
    
    prompt = f"""
Write a helpful, informative blog post about: "{topic}"

BASED ON THESE SOURCES:
{sources_text}

RESEARCH SUMMARY:
{research.get('research_summary', '')}

REQUIREMENTS:
1. Write approx 5000 words
2. Use a friendly, expert tone
3. Include practical advice or actionable tips
4. Cite sources inline like [Source: Title]
5. End with a conclusion
6. Start with '# Title' on first line

Write the complete post now:
"""
    
    response = ollama.chat(
        model="llama3.1",
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    content = response['message']['content']
    
    # Estrai titolo
    lines = content.split('\n')
    title = topic
    for line in lines:
        if line.startswith('# '):
            title = line[2:].strip()
            break
    
    post = {
        "draft_post": {
            "title": title,
            "content": content,
            "topic": topic,
            "sources": research.get('sources', []),
            "error": False
        }
    }
    
    print(f"✅ Post generated: '{title}'")
    print("="*50)
    
    return post
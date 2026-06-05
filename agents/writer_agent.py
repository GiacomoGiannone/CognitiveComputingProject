# agents/writer_agent.py
import ollama

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
        sources_text += f"""
SOURCE {i+1}:
Title: {doc.get('title', 'No title')}
URL: {doc.get('url', 'No URL')}
Content:
{doc.get('content', '')[:1500]}
---
"""
    
    prompt = f"""
Write a helpful, informative blog post about: "{topic}"

BASED ON THESE SOURCES:
{sources_text}

RESEARCH SUMMARY:
{research.get('research_summary', '')}

REQUIREMENTS:
1. Write approx 1000-1500 words
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
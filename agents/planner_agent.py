# agents/planner_agent.py
import ollama
import json

def planner_agent(state):
    """Planner agent - genera piano editoriale e seleziona il primo topic"""
    
    print("\n" + "="*50)
    print("📋 PLANNER AGENT STARTING")
    print("="*50)
    
    # Recupera il dominio
    domain = state.get('blog_domain', 'General Blog')
    
    # Recupera topic già coperti dal KG
    covered_topics = []
    kg = state.get('kg_manager')
    if kg:
        try:
            result = kg.query("MATCH (p:Post)-[:COVERS]->(t:Topic) RETURN DISTINCT t.name as topic")
            covered_topics = [r['topic'] for r in result]
            print(f"📚 Topics already covered: {covered_topics}")
        except Exception as e:
            print(f"⚠️ Could not query KG: {e}")
    
    # Prompt per generare topic
    prompt = f"""
    Domain: {domain}
    
    {'Already covered topics: ' + ', '.join(covered_topics) if covered_topics else 'No topics covered yet.'}
    
    Generate a list of 5 blog post topics for this domain.
    Avoid topics already covered.
    
    Return ONLY a JSON array in this exact format:
    ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"]
    
    Make topics specific and concrete, not too broad.
    """
    
    # Chiamata Ollama corretta
    response = ollama.chat(
        model="llama3.1",
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    # Parsing dei topic
    try:
        # Cerca array JSON nella risposta
        content = response['message']['content']
        # Trova la prima lista JSON
        import re
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            topics = json.loads(match.group())
        else:
            topics = ["Manutenzione barca a vela", "Tecniche di navigazione", 
                     "Sicurezza in mare", "Regolamenti nautici", "Corsi di vela"]
    except:
        # Fallback
        topics = ["Manutenzione barca a vela", "Tecniche di navigazione", 
                 "Sicurezza in mare", "Regolamenti nautici", "Corsi di vela"]
    
    # Seleziona il primo topic come current_topic
    current_topic = topics[0] if topics else "Manutenzione barca a vela"
    
    editorial_plan = f"""
    📅 EDITORIAL PLAN for {domain}:
    
    1. {topics[0]} (next post)
    2. {topics[1]}
    3. {topics[2]}
    4. {topics[3]}
    5. {topics[4]}
    
    Justification: Topics are ordered from most practical/urgent to more advanced.
    """
    
    print(f"✅ Selected topic for research: {current_topic}")
    print("="*50)
    
    return {
        "editorial_plan": editorial_plan,
        "current_topic": current_topic,  # IMPORTANTE: imposta il topic corrente
        "all_topics": topics
    }
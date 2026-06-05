# agents/planner_agent.py
import ollama
import json
import os
from datetime import datetime

def planner_agent(state):
    """Planner agent - genera piano editoriale e seleziona il primo topic"""
    
    print("\n" + "="*50)
    print("📋 PLANNER AGENT STARTING")
    print("="*50)
    
    # Recupera il dominio (forza sport per questo progetto)
    domain = state.get('blog_domain', 'Sport')
    domain_label = "Sport" if "sport" not in str(domain).lower() else domain

    # Memoria persistente per piani editoriali
    memory_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "editorial_memory.json")
    )

    def load_memory(path: str) -> dict:
        if not os.path.exists(path):
            return {"domain": "sport", "plans": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"domain": "sport", "plans": []}

    def save_memory(path: str, data: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)

    memory = load_memory(memory_path)
    previous_topics = set()
    for plan in memory.get("plans", []):
        for t in plan.get("topics", []):
            if isinstance(t, str) and t.strip():
                previous_topics.add(t.strip())
    
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
    
    # Prompt per generare topic sportivi
    prompt = f"""
    Domain: {domain_label}
    This is a SPORTS blog. Generate ONLY sports-related topics.
    
    {'Already covered topics: ' + ', '.join(covered_topics) if covered_topics else 'No topics covered yet.'}
    {'Previously suggested topics: ' + ', '.join(sorted(previous_topics)) if previous_topics else 'No previous planner memory.'}
    
    Generate a list of 5 blog post topics for this domain.
    Avoid topics already covered.
    Avoid topics already suggested in previous runs.
    
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
            topics = [
                "Piani di allenamento per runner principianti",
                "Nutrizione sportiva per sport di endurance",
                "Prevenzione infortuni nel calcio dilettantistico",
                "Tecnologia e analytics nel basket moderno",
                "Recupero muscolare e sonno per atleti"
            ]
    except:
        # Fallback
        topics = [
            "Piani di allenamento per runner principianti",
            "Nutrizione sportiva per sport di endurance",
            "Prevenzione infortuni nel calcio dilettantistico",
            "Tecnologia e analytics nel basket moderno",
            "Recupero muscolare e sonno per atleti"
        ]

    # Filtra duplicati e topic già coperti/memorizzati
    avoid_topics = set(covered_topics) | previous_topics
    cleaned = []
    seen = set()
    for t in topics:
        if not isinstance(t, str):
            continue
        topic = t.strip()
        if not topic:
            continue
        if topic in avoid_topics:
            continue
        key = topic.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(topic)

    fallback_pool = [
        "Allenamento funzionale per sport di squadra",
        "Strategie mentali per la performance sportiva",
        "Come scegliere l'attrezzatura per il ciclismo",
        "Tecniche di recupero per triatleti",
        "Analisi tattica nel calcio moderno",
        "Riscaldamento efficace per sport ad alta intensita",
        "Prevenzione infortuni per runners",
        "Guida alla forza per atleti di volley",
        "Nutrizione pre-gara per sport di resistenza",
        "Tecnologia wearable nello sport"
    ]

    for t in fallback_pool:
        if len(cleaned) >= 5:
            break
        if t in avoid_topics:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(t)

    topics = cleaned[:5]
    
    # Seleziona il primo topic come current_topic
    current_topic = topics[0] if topics else "Allenamento funzionale per sport di squadra"
    
    editorial_plan = f"""
    📅 EDITORIAL PLAN for {domain}:
    
    1. {topics[0]} (next post)
    2. {topics[1]}
    3. {topics[2]}
    4. {topics[3]}
    5. {topics[4]}
    
    Justification: Topics are ordered from most practical/urgent to more advanced.
    """
    
    # Salva piano in memoria persistente
    memory["domain"] = "sport"
    memory.setdefault("plans", []).append({
        "created_at": datetime.utcnow().isoformat() + "Z",
        "topics": topics
    })
    save_memory(memory_path, memory)

    print(f"✅ Selected topic for research: {current_topic}")
    print(f"💾 Saved editorial plan to: {memory_path}")
    print("="*50)
    
    return {
        "editorial_plan": editorial_plan,
        "current_topic": current_topic,  # IMPORTANTE: imposta il topic corrente
        "all_topics": topics
    }
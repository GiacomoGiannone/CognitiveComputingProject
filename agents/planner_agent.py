# agents/planner_agent.py
import ollama
import json
import os
from datetime import datetime
from tools.entity_extractor import extract_topics_from_title

def planner_agent(state):
    """Planner agent - genera piano editoriale e seleziona il primo topic"""
    
    print("\n" + "="*50)
    print("📋 PLANNER AGENT STARTING")
    print("="*50)
    
    # Recupera il dominio
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
            json.dump(data, f, indent=2, ensure_ascii=False)

    memory = load_memory(memory_path)
    
    # Normalizza tutti i piani
    for plan in memory.get("plans", []):
        topics_normalized = []
        for idx, item in enumerate(plan.get("topics", [])):
            if isinstance(item, str):
                topics_normalized.append({
                    "index": idx,
                    "topic": item,
                    "finished": False,
                    "finished_at": None
                })
            elif isinstance(item, dict):
                topics_normalized.append({
                    "index": item.get("index", idx),
                    "topic": item.get("topic") or item.get("name", ""),
                    "finished": item.get("finished", False),
                    "finished_at": item.get("finished_at")
                })
        plan["topics"] = topics_normalized
        plan.setdefault("finished", False)
        plan.setdefault("last_topic_index", -1)
    
    # CERCA IL PIANO ATTIVO (primo piano con finished=False)
    active_plan = None
    active_plan_idx = -1
    
    for idx, plan in enumerate(memory.get("plans", [])):
        if not plan.get("finished", False):
            active_plan = plan
            active_plan_idx = idx
            print(f"📋 Found active plan (index {idx})")
            break
    
    # Se non c'è piano attivo, creane uno nuovo
    if active_plan is None:
        print("📋 No active plan found, creating new plan...")
        
        #  Recupera SOLO i topic coperti di recente (nell'ultimo giorno)
        covered_topics = []
        kg = state.get('kg_manager')
        if kg:
            try:
                covered_topics = kg.get_recently_covered_topics(days=1)
                print(f"⏰ Topics covered in LAST 24 HOURS (cooldown): {covered_topics}")
            except Exception as e:
                print(f"⚠️ Could not query KG: {e}")
        
        # Raccolgi tutti i topic già usati
        previous_topics = set(covered_topics)
        for plan in memory.get("plans", []):
            for topic_item in plan.get("topics", []):
                if topic_item.get("topic"):
                    previous_topics.add(topic_item.get("topic"))
        
        avoid_list = previous_topics
        avoid_text = ", ".join(sorted(avoid_list)) if avoid_list else "None"
        
        prompt = f"""
        Domain: {domain_label}
        This is a SPORTS blog. Generate ONLY sports-related topics.
        
        Topics to AVOID (already covered or previously suggested):
        {avoid_text}
        
        Generate a list of 5 NEW blog post topics for this domain.
        Do NOT repeat any of the topics above.
        
        Return ONLY a JSON object with exactly two keys:
        - "topics": a list of 5 strings representing the new topics.
        - "justification": a string explaining the editorial rationale behind these topics.
        
        Example format:
        {{
            "topics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"],
            "justification": "This plan covers key fundamentals and advanced topics to capture beginner to expert readers sequentially."
        }}
        
        Make topics specific and concrete, not too broad.
        Return ONLY the JSON object, nothing else.
        """
        
        response = ollama.chat(
            model="llama3.1",
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Parsing dei topic e della giustificazione
        justification = "Sequential order based on plan creation."
        try:
            import re
            content = response['message']['content']
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                topics = data.get("topics", [])
                justification = data.get("justification", "Sequential order based on plan creation.")
            else:
                topics = [
                    "Allenamento pliometrico per sport di squadra",
                    "Gestione del carico per atleti amatoriali",
                    "Tecniche di respirazione per il nuoto",
                    "Preparazione mentale per competizioni",
                    "Recupero attivo dopo l'allenamento"
                ]
        except Exception as e:
            print(f"⚠️ Error parsing plan JSON: {e}")
            topics = [
                "Allenamento pliometrico per sport di squadra",
                "Gestione del carico per atleti amatoriali",
                "Tecniche di respirazione per il nuoto",
                "Preparazione mentale per competizioni",
                "Recupero attivo dopo l'allenamento"
            ]
        
        # Filtra duplicati
        cleaned = []
        seen = set()
        for t in topics:
            if not isinstance(t, str):
                continue
            topic = t.strip()
            if not topic or topic in avoid_list:
                continue
            key = topic.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(topic)
        
        topics = cleaned[:5]
        
        if len(topics) < 5:
            fallback = [
                "Mobilità articolare per sportivi",
                "Periodizzazione dell'allenamento",
                "Idratazione durante l'attività sportiva",
                "Tecnica di corsa per principianti",
                "Stretching dinamico pre-allenamento"
            ]
            for fb in fallback:
                if len(topics) >= 5:
                    break
                if fb not in avoid_list and fb.lower() not in seen:
                    topics.append(fb)
                    seen.add(fb.lower())
        
        # Crea nuovo piano
        active_plan = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "finished": False,
            "last_topic_index": -1,
            "justification": justification,
            "topics": [
                {
                    "index": idx,
                    "topic": topic_name,
                    "finished": False,
                    "finished_at": None,
                }
                for idx, topic_name in enumerate(topics)
            ]
        }
        memory.setdefault("plans", []).append(active_plan)
        save_memory(memory_path, memory)
        
        print("\n✨ Generated NEW editorial plan:")
        for idx, topic_name in enumerate(topics, 1):
            print(f"  {idx}. {topic_name}")
        print(f"  Justification: {justification}")
    
    # ORA DETERMINA IL PROSSIMO TOPIC DAL PIANO ATTIVO
    # ALGORITMO: prendi l'indice dell'ultimo topic e incrementa di 1
    
    last_index = active_plan.get("last_topic_index", -1)
    next_index = last_index + 1
    
    topics_list = active_plan.get("topics", [])
    
    # Verifica se il piano è completo
    if next_index >= len(topics_list):
        # Piano completato! Marca come finished e cerca un nuovo piano
        print(f"\n✅ Plan completed! Marking as finished.")
        active_plan["finished"] = True
        active_plan["finished_at"] = datetime.utcnow().isoformat() + "Z"
        save_memory(memory_path, memory)
        
        # Ricorsione per trovare/prossimo piano
        return planner_agent(state)
    
    # Prendi il topic all'indice next_index
    next_topic_item = topics_list[next_index]
    current_topic = next_topic_item.get("topic")
    
    # Calcola i topic rimanenti
    remaining_topics = [
        t.get("topic") for t in topics_list[next_index:]
        if t.get("topic") and not t.get("finished", False)
    ]
    
    # Stampa stato
    print(f"\n📋 Active Plan Status:")
    print(f"   Plan created: {active_plan.get('created_at', 'Unknown')}")
    print(f"   Justification: {active_plan.get('justification', 'Sequential order based on plan creation.')}")
    print(f"   Last completed index: {last_index}")
    print(f"   Next index: {next_index}")
    print(f"   Total topics: {len(topics_list)}")
    
    print(f"\n🗂️ Topics in plan:")
    for idx, topic_item in enumerate(topics_list):
        if idx < next_index:
            status = "✅ DONE"
        elif idx == next_index:
            status = "🎯 CURRENT"
        else:
            status = "⏳ PENDING"
        print(f"   {status}: {topic_item.get('topic')}")
    
    editorial_plan = f"""
    📅 EDITORIAL PLAN for {domain}:
    
    Current post ({next_index + 1}/{len(topics_list)}): {current_topic}
    Remaining topics: {', '.join(remaining_topics[1:5]) if len(remaining_topics) > 1 else 'None'}
    
    Justification: {active_plan.get('justification', 'Sequential order based on plan creation.')}
    """
    
    print(f"\n✅ Selected topic for research: {current_topic}")
    print("="*50)
    
    # Estrai i topic generici dal titolo specifico
    print(f"\n🔧 Extracting generic topics for Knowledge Graph...")
    domain = state.get('blog_domain')
    extracted_topics = extract_topics_from_title(current_topic, domain)
    #print(f"✅ Generic topics extracted: {extracted_topics}")
    
    return {
        "editorial_plan": editorial_plan,
        "current_topic": current_topic,
        "all_topics": remaining_topics,
        "extracted_graph_topics": extracted_topics 
    }
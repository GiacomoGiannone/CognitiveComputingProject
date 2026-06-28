# agents/planner_agent.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langsmith import traceable
import json
import os
from datetime import datetime
from tools.entity_extractor import extract_topics_from_title

@traceable(name="PlannerAgent", run_type="chain", tags=["agent", "planner"])
def planner_agent(state):
    """Planner agent - genera piano editoriale e seleziona il primo topic"""
    
    print("\n" + "="*50)
    print("PLANNER AGENT STARTING")
    print("="*50)
    
    # Recupera il dominio
    # Se il dominio non è specificato, usa "Sport" come default
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
    
    # CERCA IL PIANO ATTIVO (primo piano con finished=False)
    active_plan = None
    active_plan_idx = -1
    
    for idx, plan in enumerate(memory.get("plans", [])):
        if not plan.get("finished", False):
            active_plan = plan
            active_plan_idx = idx
            print(f" Found active plan (index {idx})")
            break
    
    # Se non c'è piano attivo, creane uno nuovo
    if active_plan is None:
        print(" No active plan found, creating new plan...")
        
        #  Recupera SOLO i topic coperti di recente (nell'ultimo giorno)
        covered_topics = []
        kg = state.get('kg_manager')
        if kg:
            try:
                covered_topics = kg.get_recently_covered_topics(days=1)
                print(f" Topics covered in LAST 24 HOURS (cooldown): {covered_topics}")
            except Exception as e:
                print(f" No topic to mark as completed")
        
        # Raccogli tutti i topic già usati nelle ultime 24 ore (cooldown)
        previous_topics = set(covered_topics)
        now = datetime.utcnow()
        for plan in memory.get("plans", []):
            # Controlla se il piano è stato creato nelle ultime 24 ore
            plan_created_at_str = plan.get("created_at")
            plan_in_last_24h = False
            if plan_created_at_str:
                try:
                    clean_ts = plan_created_at_str.replace("Z", "")
                    plan_time = datetime.fromisoformat(clean_ts)
                    if (now - plan_time).total_seconds() <= 86400:
                        plan_in_last_24h = True
                except Exception as e:
                    print(f" Error parsing plan created_at: {e}")

            for topic_item in plan.get("topics", []):
                topic_name = topic_item.get("topic")
                if not topic_name:
                    continue
                
                # Se il piano è stato creato nelle ultime 24 ore, escludi tutti i suoi topic
                if plan_in_last_24h:
                    previous_topics.add(topic_name)
                    continue
                
                # Altrimenti, controlla se il topic specifico è stato completato nelle ultime 24 ore
                finished_at_str = topic_item.get("finished_at")
                if finished_at_str:
                    try:
                        clean_ts = finished_at_str.replace("Z", "")
                        finish_time = datetime.fromisoformat(clean_ts)
                        if (now - finish_time).total_seconds() <= 86400:
                            previous_topics.add(topic_name)
                    except Exception as e:
                        print(f" Error parsing topic finished_at: {e}")
        
        avoid_list = previous_topics
        #crea una stringa con i topic da evitare per il prompt
        avoid_text = ", ".join(sorted(avoid_list)) if avoid_list else "None"
        
        prompt = f"""
        Domain: {domain_label}
        This is a SPORTS blog. Generate ONLY sports-related topics.
        
        Topics to AVOID (already covered or previously suggested):
        {avoid_text}
        
        Generate a list of 5 NEW blog post topics for this domain.
        Do NOT repeat any of the topics above.
        
        Return ONLY a JSON object with exactly two keys:
        - "topics": a list of 5 strings representing the new topics, ordered by publishing priority.
        - "justification": a string explaining both the editorial rationale behind the topic selection AND the reason for their specific ordering.
        
        Example format:
        {{
            "topics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"],
            "justification": "Topics progress from foundational concepts (Topic 1) to advanced techniques (Topic 5), allowing readers to build knowledge incrementally across the series."
        }}
        
        Make topics specific and concrete, not too broad.
        Return ONLY the JSON object, nothing else.
        """
        
        llm = ChatOllama(model="llama3.1", temperature=0.7)
        response = llm.invoke([HumanMessage(content=prompt)])
        
        """Parsing dei topic e della giustificazione"""

        # Una giustificazione di default se il parsing fallisce, per evitare crash
        justification = "Sequential order based on plan creation." 
        try:
            import re
            content = response.content
            #re.DOTALL permette di fare match su più linee
            #il pattern \{.*?\} cerca il primo oggetto JSON tra parentesi graffe
            #il pattern \{.*\} cerca l'ultimo oggetto JSON tra parentesi graffe
            #come mai lo facciamo? Perché a volte il modello può generare testo extra prima o dopo l'oggetto JSON, 
            #quindi cerchiamo di estrarre solo l'oggetto JSON valido
            #ne facciamo due tentativi: prima cerchiamo il primo match non greedy, se non lo troviamo, cerchiamo l'ultimo match greedy
            #il match greedy prende tutto il contenuto tra la prima e l'ultima parentesi graffa, quindi potrebbe includere testo extra, ma almeno ci dà un oggetto JSON valido
            #il match non greedy e' intelligente perché prende solo il primo oggetto JSON, ma se il modello genera testo extra prima o dopo, potrebbe non funzionare
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                topics = data.get("topics", [])
                justification = data.get("justification", justification)
            else:
                topics = [
                    "Allenamento pliometrico per sport di squadra",
                    "Gestione del carico per atleti amatoriali",
                    "Tecniche di respirazione per il nuoto",
                    "Preparazione mentale per competizioni",
                    "Recupero attivo dopo l'allenamento"
                ]
        except Exception as e:
            print(f" Error parsing plan JSON: {e}")
            topics = [
                "Allenamento pliometrico per sport di squadra",
                "Gestione del carico per atleti amatoriali",
                "Tecniche di respirazione per il nuoto",
                "Preparazione mentale per competizioni",
                "Recupero attivo dopo l'allenamento"
            ]
        
        # Filtra duplicati
        #potrebbero esserci duplicati o topic vuoti, quindi li filtriamo
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
        
        print("\n Generated NEW editorial plan:")
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
        print(f"\n Plan completed! Marking as finished.")
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
    
    print(f"\n Topics in plan:")
    for idx, topic_item in enumerate(topics_list):
        if idx < next_index:
            status = " DONE"
        elif idx == next_index:
            status = " CURRENT"
        else:
            status = " PENDING"
        print(f"   {status}: {topic_item.get('topic')}")
    
    editorial_plan = f"""
     EDITORIAL PLAN for {domain}:
    
    Current post ({next_index + 1}/{len(topics_list)}): {current_topic}
    Remaining topics: {', '.join(remaining_topics[1:5]) if len(remaining_topics) > 1 else 'None'}
    
    Justification: {active_plan.get('justification', 'Sequential order based on plan creation.')}
    """
    
    print(f"\n Selected topic for research: {current_topic}")
    print("="*50)
    
    # Estrai i topic generici dal titolo specifico
    print(f"\n🔧 Extracting generic topics for Knowledge Graph...")
    domain = state.get('blog_domain')
    extracted_topics = extract_topics_from_title(current_topic, domain)
    #print(f" Generic topics extracted: {extracted_topics}")
    
    #editorial_plan non viene utilizzata?
    return {
        "editorial_plan": editorial_plan,
        "current_topic": current_topic,
        "all_topics": remaining_topics,
        "extracted_graph_topics": extracted_topics 
    }
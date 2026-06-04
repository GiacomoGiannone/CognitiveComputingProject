from datetime import datetime
from difflib import SequenceMatcher

def find_matching_topic(kg, topic_name, threshold=0.6):
    """Trova un topic esistente che corrisponde al nome"""
    topics = kg.get_nodes_by_type("Topic")
    
    topic_name_lower = topic_name.lower()
    best_match = None
    best_score = 0
    
    for node_id, node_data in topics.items():
        name = node_data.get("properties", {}).get("name", "")
        if not name:
            continue
        
        # Similarità tra stringhe
        ratio = SequenceMatcher(None, topic_name_lower, name.lower()).ratio()
        
        # Bonus se una frase contiene l'altra
        if topic_name_lower in name.lower() or name.lower() in topic_name_lower:
            ratio = max(ratio, 0.7)
        
        # Bonus per parole chiave importanti
        keywords = ["calciomercato", "allenamento", "nba", "serie a", "inter", "juventus", "milan"]
        for kw in keywords:
            if kw in topic_name_lower and kw in name.lower():
                ratio += 0.1
        
        if ratio > best_score:
            best_score = ratio
            best_match = node_id
    
    return best_match if best_score >= threshold else None

def kg_update_node(state):
    """Aggiorna il KG con il nuovo post approvato"""
    kg = state.get("kg")
    
    if kg is None:
        print("[kg_update] KG non trovato")
        return state
    
    # Solo se il feedback è positivo
    if state.get("content_feedback") != "yes":
        print("[kg_update] Feedback non positivo, salto l'aggiornamento")
        return state
    
    topic_name = state.get("chosen_topic", "")
    if not topic_name:
        print("[kg_update] Nessun topic specificato")
        return state
    
    print(f"[kg_update] Cerco/Creo topic per: '{topic_name}'")
    
    # Cerca topic esistente
    existing_topic_id = find_matching_topic(kg, topic_name)
    
    if existing_topic_id:
        topic_id = existing_topic_id
        topic_data = kg.get_node(topic_id)
        print(f"[kg_update] Usando topic esistente: '{topic_data['properties'].get('name')}' (ID: {topic_id})")
    else:
        # Crea nuovo topic
        max_topic_num = 0
        for node_id in kg.data["nodes"]:
            if node_id.startswith("topic_"):
                try:
                    num = int(node_id.split("_")[1])
                    max_topic_num = max(max_topic_num, num)
                except:
                    pass
        
        topic_id = f"topic_{max_topic_num + 1}"
        kg.add_node(topic_id, "Topic", {"name": topic_name})
        print(f"[kg_update] Creato nuovo topic: '{topic_name}' (ID: {topic_id})")
    
    # Crea ID univoco per il post
    max_post_num = 0
    for node_id in kg.data["nodes"]:
        if node_id.startswith("post_"):
            try:
                num = int(node_id.split("_")[1])
                max_post_num = max(max_post_num, num)
            except:
                pass
    
    new_post_id = f"post_{max_post_num + 1}"
    
    # Aggiungi il post
    kg.add_node(
        new_post_id,
        "Post",
        {
            "content": state["created_content"],
            "created_at": datetime.now().isoformat()
        }
    )
    
    # Crea la relazione
    kg.add_relationship(new_post_id, "COVERS", topic_id)
    
    print(f"[kg_update] Aggiunto {new_post_id} -> COVERS -> {topic_id}")
    
    return state
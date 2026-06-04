from langchain.tools import tool
from difflib import SequenceMatcher
import datetime

def get_topics_already_covered(kg, topic_id):
    posts = kg.get_posts_about_topic(topic_id)
    return len(posts) > 0

def get_N_most_recent_topics(kg, N=5):
    posts = kg.get_nodes_by_type("Post")
    sorted_posts = sorted(posts.items(), key=lambda x: x[1]["properties"].get("date", "1970-01-01"), reverse=True)

    topics = []
    seen_topic_ids = set()

    for post_id, post_data in sorted_posts:
        if len(topics) >= N:
            break
        
        topic_ids = kg.get_connected_nodes(post_id, relation="COVERS")
        for topic_id in topic_ids:
            if len(topics) >= N:
                break
            if topic_id in seen_topic_ids:
                continue

            topic_data = kg.get_node(topic_id)
            if not topic_data or topic_data.get("type") != "Topic":
                continue

            topics.append({"id": topic_id, **topic_data})
            seen_topic_ids.add(topic_id)

    return topics

def find_similar_topic(kg, topic_name, threshold=0.6):
    """Cerca topic simili nel KG basandosi sul nome"""
    topics = kg.get_nodes_by_type("Topic")
    best_match = None
    best_ratio = 0
    
    topic_name_lower = topic_name.lower()
    
    for node_id, node_data in topics.items():
        name = node_data.get("properties", {}).get("name", "")
        if not name:
            continue
            
        # Calcola similarità tra stringhe
        ratio = SequenceMatcher(None, topic_name_lower, name.lower()).ratio()
        
        # Bonus se una contiene l'altra
        if topic_name_lower in name.lower() or name.lower() in topic_name_lower:
            ratio = max(ratio, 0.7)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = (node_id, node_data)
    
    if best_ratio >= threshold:
        return best_match[0], best_match[1]
    return None, None

@tool
def add_edge_to_kg(state, topic_name, post_content):
    """
    Tool per aggiungere un nuovo post che copre un topic al KG.
    Cerca prima se esiste già un topic simile, altrimenti ne crea uno nuovo.
    
    Args:
        state: Lo stato dell'agente
        topic_name: Il nome del topic (stringa)
        post_content: Il contenuto del post da aggiungere
    """
    kg = state.get("kg")
    
    if kg is None:
        print("[add_edge_to_kg] KG non trovato nello stato")
        return state
    
    print(f"[add_edge_to_kg] Cerco topic: '{topic_name}'")
    
    # 1. Cerca un topic esistente simile
    existing_topic_id, existing_topic = find_similar_topic(kg, topic_name)
    
    if existing_topic_id:
        topic_id = existing_topic_id
        topic_node = existing_topic
        print(f"[add_edge_to_kg] Trovato topic esistente: '{topic_node['properties'].get('name')}' (ID: {topic_id})")
    else:
        # 2. Crea nuovo topic
        # Trova il massimo ID numerico esistente per i topic
        max_topic_num = 0
        for node_id in kg.data["nodes"]:
            if node_id.startswith("topic_"):
                try:
                    num = int(node_id.split("_")[1])
                    max_topic_num = max(max_topic_num, num)
                except:
                    pass
        
        topic_id = f"topic_{max_topic_num + 1}"
        kg.add_node(
            topic_id,
            "Topic",
            {"name": topic_name}
        )
        print(f"[add_edge_to_kg] Creato nuovo topic: '{topic_name}' (ID: {topic_id})")
    
    # 3. Trova ID univoco per il post
    max_post_num = 0
    for node_id in kg.data["nodes"]:
        if node_id.startswith("post_"):
            try:
                num = int(node_id.split("_")[1])
                max_post_num = max(max_post_num, num)
            except:
                pass
    
    new_post_id = f"post_{max_post_num + 1}"
    
    # 4. Aggiungi il post
    kg.add_node(
        new_post_id,
        "Post",
        {
            "content": post_content,
            "created_at": datetime.now().isoformat()  # Aggiungi timestamp
        }
    )
    
    # 5. Aggiungi la relazione
    kg.add_relationship(new_post_id, "COVERS", topic_id)
    
    print(f"[add_edge_to_kg] Post aggiunto: {new_post_id} -> COVERS -> {topic_id}")
    
    return state
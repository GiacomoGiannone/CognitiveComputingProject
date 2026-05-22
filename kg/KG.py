
import json
from datetime import datetime 
from typing import Dict, Any
import os

class KG():
    def __init__(self, path = "kg.json"):
        self.path = path
        self.data = {
            "nodes" : {},
            "relationships" :[]
        }
        #che struttura ha self.data?
        """{
            "nodes": {
                "node_1": {
                "type": "Topic",
                "properties": {
                    "nome": "Intelligenza Artificiale",
                    "difficoltà": "alta"
                },
                "created_at": "2026-05-21T15:30:00.123456"
                },
                "node_2": {
                "type": "Post",
                "properties": {
                    "titolo": "Introduzione ai LLM"
                },
                "created_at": "2026-05-21T15:35:00.123456"
                }
            },
            "relationships": [
                {
                "source": "node_2",
                "relation": "COVERS",
                "target": "node_1"
                }
            ]
            }"""

        self.load()

    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        #Aggiunge un nodo con validazione del tipo
        valid_types = {"Topic", "Post", "Entity", "Event", "Tag", "Resource"}
        if node_type not in valid_types:
            raise ValueError(f"node_type must be one of {valid_types}")
        
        self.data["nodes"][node_id] = {
            "type": node_type,
            "properties": properties or {}, #le properties salvano titolo, URL, contenuto, punteggio di importanza,  ecc.
            "created_at": datetime.now().isoformat()
        }
        self.save()
    
    def add_relationship(self, source: str, relation: str, target: str):
        #Aggiunge una relazione con validazione
        valid_relations = {"COVERS", "RELATED_TO", "REQUIRES", "MENTIONS", "HAS_TAG", "LOCATED_IN"}
        if relation not in valid_relations:
            raise ValueError(f"relation must be one of {valid_relations}")
        
        # Verifica che source e target esistano
        if source not in self.data["nodes"]:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self.data["nodes"]:
            raise ValueError(f"Target node '{target}' not found")
        
        self.data["relationships"].append({
            "source": source,
            "relation": relation,
            "target": target
        })
        self.save()

    def get_node(self, node_id: str):
        return self.data["nodes"].get(node_id)
    
    def get_nodes_by_type(self, node_type: str):
        output = {}
        for node_id, node in self.data["nodes"].items():
            if node["type"] == node_type:
                output[node_id] = node
        return output

    def get_relationships(self, node_id):
        #Restituisce tutte le relazioni che coinvolgono il nodo.
        output = []
        for rel in self.data["relationships"]:
            if rel["source"] == node_id or rel["target"] == node_id:
                output.append(rel)
        return output

    def get_connected_nodes(self, node_id, relation=None):
        #Trova gli ID dei nodi connessi. Opzionalmente filtra per tipo di relazione
        connected = []
        for rel in self.data["relationships"]:
            if relation and rel["relation"] != relation:
                continue
            
            if rel["source"] == node_id:
                connected.append(rel["target"])
            elif rel["target"] == node_id:
                connected.append(rel["source"])
                
        # Rimuove eventuali duplicati
        return list(set(connected))

    def get_posts_about_topic(self, topic_id):
        #Restituisce i nodi di tipo Post associati a un certo Topic.
        connected_ids = self.get_connected_nodes(topic_id)
        posts = []
        for n_id in connected_ids:
            node_data = self.get_node(n_id)
            if node_data and node_data["type"] == "Post":
                posts.append({"id": n_id, **node_data})
        return posts
    
    def save(self):
        #Salva su file JSON
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def load(self):
        #Carica il grafo dal file JSON se esiste.
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {"nodes": {}, "relationships": []}
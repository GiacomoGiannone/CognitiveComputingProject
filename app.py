import os

import google.genai as genai

from dotenv import load_dotenv
import ollama

from graph.workflow import create_blog_workflow
from kg.neo4j_manager import Neo4jManager

# populate_kg.py

def populate_initial_graph(kg_manager: Neo4jManager, blog_domain: str):
    """Popola il grafo con topic iniziali"""
    
    # Topic base del dominio
    base_topics = {
        "Vela": "Sport e attività nautiche",
        "Manutenzione barca": "Cura e manutenzione delle imbarcazioni",
        "Tecniche di navigazione": "Metodi e tecniche per navigare",
        "Sicurezza in mare": "Procedure e attrezzature di sicurezza",
        "Regolamenti nautici": "Normative e leggi sulla navigazione",
        "Corsi di vela": "Formazione e certificazioni",
        "Apparecchiature elettroniche": "Strumenti e tecnologie di bordo"
    }
    
    # Aggiungi topic base
    for topic, description in base_topics.items():
        kg_manager.add_topic(topic, description)
    
    # Aggiungi relazioni tra topic
    relations = [
        ("Vela", "Tecniche di navigazione"),
        ("Vela", "Manutenzione barca"),
        ("Manutenzione barca", "Apparecchiature elettroniche"),
        ("Sicurezza in mare", "Regolamenti nautici"),
        ("Tecniche di navigazione", "Sicurezza in mare")
    ]
    
    for t1, t2 in relations:
        kg_manager.add_topic_relation(t1, t2)
    
    # Aggiungi post esistenti (esempio)
    existing_posts = [
        {
            "title": "Manutenzione base del winch",
            "content": "...",
            "topics": ["Manutenzione barca", "Vela"],
            "sources": ["https://sailingexample.com/winch-maintenance"]
        },
        {
            "title": "Come leggere le carte nautiche",
            "content": "...",
            "topics": ["Tecniche di navigazione", "Sicurezza in mare"],
            "sources": ["https://navigation-guide.com/charts"]
        }
    ]
    
    for post in existing_posts:
        kg_manager.add_post(
            title=post["title"],
            content=post["content"],
            topics=post["topics"],
            sources=post["sources"]
        )
    
    print(f"✅ Grafo popolato con {len(base_topics)} topic e {len(existing_posts)} post")

if __name__ == "__main__":
    load_dotenv()

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_uri:
        raise ValueError("NEO4J_URI is missing. Set it to something like bolt://localhost:7687")

    kg = Neo4jManager(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )
    populate_initial_graph(kg, "Vela")
    app = create_blog_workflow(kg)
    initial_state = {
        "blog_domain": "Vela e nautica",
        "kg_manager": kg,
        "current_topic": "Manutenzione preventiva winch",
        "max_post_length": 1500,
        "iteration": 0,
        "max_iterations": 3
    }
    for output in app.stream(initial_state):
        print(f"Step completed: {output}")
    kg.close()
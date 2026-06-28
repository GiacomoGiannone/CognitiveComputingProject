# neo4j_manager.py
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from langchain.tools import tool

# Global reference to the Neo4jManager instance for tool calling
_global_neo4j_manager = None

class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_schema()
        global _global_neo4j_manager
        _global_neo4j_manager = self

    
    #metodo privato per inizializzare lo schema del database
    def _init_schema(self):
        """Inizializza i constraint e indici"""
        #apriamo una sessione con with, in tal modo la sessione viene chiusa automaticamente al termine del blocco
        with self.driver.session() as session:
            # Crea constraint
            session.run("CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE") #topic name deve essere unico
            session.run("CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE") #post id deve essere unico
            session.run("CREATE CONSTRAINT source_url IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE") #source url deve essere unico
    
    # def add_topic(self, name: str, description: str = None) -> Dict:
    #     """Aggiunge un topic al grafo"""
    #     with self.driver.session() as session:
    #         #MERGE cerca un nodo TOPIC con quel nome. SE non esiste, lo crea. 
    #         #SET fa assegnamento della descrizione e della data di creazione
    #         result = session.run(
    #             """
    #             MERGE (t:Topic {name: $name})
    #             SET t.description = $description,
    #                 t.created_at = datetime()
    #             RETURN t.name as name, t.description as description
    #             """,
    #             name=name, description=description
    #         )
    #         return result.single().data()
    
    def add_post(self, title: str, content: str, topics: List[str], sources: List[str], claims: List[str] = None) -> str:
        """Aggiunge un post e lo connette a topics, sources e claims"""
        #genera un id random per il post
        post_id = str(uuid.uuid4())
        
        with self.driver.session() as session:
            # Crea il post
            # Usiamo CREATE al post di MERGE perche' siamo sicuri che i POST siano univoci per id, quindi non vogliamo fare MERGE su un id esistente
            session.run(
                """
                CREATE (p:Post {
                    id: $id,
                    title: $title,
                    content: $content,
                    created_at: datetime(),
                    status: 'published'
                })
                RETURN p
                """,
                id=post_id, title=title, content=content
            )
            
            # Connetti a topic
            for topic in topics:    #per ogni topic nella lista dei topics, troviamo il post, e facciamo MERGE sul topic (se non esiste lo crea), e poi creiamo la relazione COVERS tra il post e il topic
                session.run(
                    """
                    MATCH (p:Post {id: $post_id})   
                    MERGE (t:Topic {name: $topic_name})
                    CREATE (p)-[:COVERS]->(t)
                    """,
                    post_id=post_id, topic_name=topic
                )
            
            # Connetti a sources
            for source_url in sources:
                session.run(
                    """
                    MATCH (p:Post {id: $post_id})
                    MERGE (s:Source {url: $url})
                    SET s.last_cited = datetime()
                    CREATE (p)-[:CITES]->(s)
                    """,
                    post_id=post_id, url=source_url
                )
            
            # Connetti a claims
            if claims:
                for claim_text in claims:
                    session.run(
                        """
                        MATCH (p:Post {id: $post_id})
                        MERGE (c:Claim {text: $claim_text})
                        CREATE (p)-[:MAKES_CLAIM]->(c)
                        """,
                        post_id=post_id, claim_text=claim_text
                    )
        
        return post_id
    
    def get_covered_topics(self) -> List[str]:
        """Restituisce tutti i topic già coperti"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Post)-[:COVERS]->(t:Topic)
                RETURN DISTINCT t.name as topic
                """
            )
            return [record["topic"] for record in result]
    
    # def get_related_topics(self, topic: str, limit: int = 5) -> List[str]:
    #     """Trova topic correlati"""
    #     #cerchiamo i topic correlati a un topic specifico con profondita' di 1 o 2, escludendo il topic stesso, e restituiamo i nomi dei topic correlati
    #     with self.driver.session() as session:
    #           result = session.run(
    #             """
    #             MATCH (t:Topic {name: $topic})-[:RELATED_TO*1..2]-(related:Topic)
    #             WHERE t.name <> related.name
    #             RETURN DISTINCT related.name as topic
    #             LIMIT $limit
    #             """,
    #             topic=topic, limit=limit
    #         )
    #           return [record["topic"] for record in result]
    
    def get_related(self, topic: str, depth: int = 1) -> List[str]:
        """Trova topic correlati con una profondità specificata (migrato da KGQueryTool)"""
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (t:Topic {{name: $topic}})-[:RELATED_TO*1..{depth}]-(related:Topic)
                WHERE t.name <> related.name
                RETURN DISTINCT related.name as topic
                LIMIT 10
                """,
                topic=topic
            )
            return [record["topic"] for record in result]

    
    def add_topic_relation(self, topic1: str, topic2: str, relation_type: str = "RELATED_TO"):
        """Aggiunge relazione tra topic"""
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (t1:Topic {{name: $topic1}})
                MATCH (t2:Topic {{name: $topic2}})
                MERGE (t1)-[:{relation_type}]->(t2)
                """,
                topic1=topic1, topic2=topic2
            )
    
    def connect_post_to_topic(self, post_id: str, topic_name: str, relation_type: str = "COVERS"):
        """Connette un Post a un Topic esistente con relazione COVERS.
        Usato per collegare il nuovo post ai topic già presenti nel KG."""
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (p:Post {{id: $post_id}})
                MATCH (t:Topic {{name: $topic_name}})
                MERGE (p)-[:{relation_type}]->(t)
                """,
                post_id=post_id, topic_name=topic_name
            )
    
    # def get_editorial_history(self) -> List[Dict]:
    #     """Recupera lo storico editoriale"""
    #     #per ogni post, recuperiamo il titolo, la data di creazione e i topic coperti, ordinando per data di creazione decrescente
    #     with self.driver.session() as session:
    #         result = session.run(
    #             """
    #             MATCH (p:Post)-[:COVERS]->(t:Topic)
    #             RETURN p.title as title, 
    #                    p.created_at as created_at,
    #                    collect(t.name) as topics
    #             ORDER BY p.created_at DESC
    #             """
    #         )
    #         return [dict(record) for record in result]
    
    # def query(self, cypher: str, params: Dict = None) -> List[Dict]:
    #     """Esecuzione query generica"""
    #     with self.driver.session() as session:
    #         result = session.run(cypher, params or {})
    #         return [record.data() for record in result]
    
    def get_recently_covered_topics(self, days: int = 1) -> List[str]:
        """ Restituisce i topic coperti SOLO negli ultimi N giorni (cooldown)"""
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (p:Post)-[:COVERS]->(t:Topic)
                WHERE p.created_at >= datetime() - duration('P{days}D')
                RETURN DISTINCT t.name as topic
                """
            )
            return [record["topic"] for record in result]
    
    def close(self):
        self.driver.close()

    # def expand_query_with_kg(self, query: str, max_related: int = 3) -> str:
    #     """
    #     Espande una query usando il Knowledge Graph
    #     Trova topic correlati per arricchire la ricerca
    #     """
    #     try:
            
    #         # Cerca topic correlati nel KG
    #         # prima cerchiamo topic correlati alla query, confrontando la query con i nomi dei topic
    #         # poi troviamo i topic correlati a quei topic
            
    #         result = self.query("""
    #             MATCH (t:Topic)
    #             WHERE t.name CONTAINS $query OR $query CONTAINS t.name
    #             MATCH (t)-[:RELATED_TO]->(related:Topic)
    #             RETURN DISTINCT related.name as related_topic
    #             LIMIT $limit
    #         """, {"query": query.lower(), "limit": max_related})
            
    #         #aggiungi i suggerimenti alla query originale
    #         if result:
    #             related = [r['related_topic'] for r in result]
    #             expanded = f"{query}\n\nRelated topics to consider: {', '.join(related)}"
    #             print(f" KG Expanded query: '{query}' -> Added related: {related}")
    #             return expanded
            
    #     except Exception as e:
    #         print(f" KG query expansion failed: {e}")
        
    #     return query
    
    # def get_context_for_topic(self, topic: str) -> str:
    #     """
    #     Recupera contesto dal KG per un topic
    #     Usato per arricchire la generazione dei post
    #     """
    #     try:
    #         # Trova topic correlati
    #         related = self.query("""
    #             MATCH (t:Topic {name: $topic})-[:RELATED_TO*1..2]-(related:Topic)
    #             WHERE t.name <> related.name
    #             RETURN DISTINCT related.name as topic
    #             LIMIT 5
    #         """, {"topic": topic})
            
    #         related_topics = [r['topic'] for r in related] if related else []
            
    #         # Trova post precedenti sul topic
    #         previous_posts = self.query("""
    #             MATCH (p:Post)-[:COVERS]->(t:Topic {name: $topic})
    #             RETURN p.title as title, p.created_at as date
    #             ORDER BY p.created_at DESC
    #             LIMIT 3
    #         """, {"topic": topic})
            
    #         context = f"""
    #         Knowledge Graph Context for: {topic}
            
    #         Related topics: {', '.join(related_topics) if related_topics else 'None'}
            
    #         Previous posts on this topic:
    #         """
    #         for post in previous_posts:
    #             context += f"\n  - {post['title']} ({post['date']})"
            
    #         return context
            
    #     except Exception as e:
    #         print(f" KG context retrieval failed: {e}")
    #         return f"No additional context found for {topic}"


@tool
def kg_search(topic: str) -> str:
    """Search the Knowledge Graph for topics related to the given topic (or multiple topics separated by commas, e.g., 'Golf, Technique, Improvement').
    Returns related topics and their connections. Use this tool to discover
    broader context, related subjects, and previously covered topics."""
    global _global_neo4j_manager
    if _global_neo4j_manager is None:
        return "Knowledge Graph not available."
    try:
        # Dividi per virgola se vengono passati più topic contemporaneamente
        topics = [t.strip() for t in topic.split(',') if t.strip()]
        all_related = []
        for t in topics:
            related = _global_neo4j_manager.get_related(t, depth=1)
            if related:
                all_related.extend(related)
        
        # Deduplica mantenendo l'ordine
        unique_related = []
        for r in all_related:
            if r not in unique_related:
                unique_related.append(r)
                
        if not unique_related:
            return f"No related topics found for '{topic}' in the Knowledge Graph."
        return f"Topics related to '{topic}': {', '.join(unique_related)}"
    except Exception as e:
        return f"KG query error: {str(e)}"
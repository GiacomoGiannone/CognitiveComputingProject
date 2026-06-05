# neo4j_manager.py
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_schema()
    
    def _init_schema(self):
        """Inizializza i constraint e indici"""
        with self.driver.session() as session:
            # Crea constraint
            session.run("CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT source_url IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE")
    
    def add_topic(self, name: str, description: str = None) -> Dict:
        """Aggiunge un topic al grafo"""
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (t:Topic {name: $name})
                SET t.description = $description,
                    t.created_at = datetime()
                RETURN t.name as name, t.description as description
                """,
                name=name, description=description
            )
            return result.single().data()
    
    def add_post(self, title: str, content: str, topics: List[str], sources: List[str]) -> str:
        """Aggiunge un post e lo connette a topics e sources"""
        post_id = str(uuid.uuid4())
        
        with self.driver.session() as session:
            # Crea il post
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
            for topic in topics:
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
    
    def get_related_topics(self, topic: str, limit: int = 5) -> List[str]:
        """Trova topic correlati"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Topic {name: $topic})-[:RELATED_TO*1..2]-(related:Topic)
                WHERE t.name <> related.name
                RETURN DISTINCT related.name as topic
                LIMIT $limit
                """,
                topic=topic, limit=limit
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
    
    def get_editorial_history(self) -> List[Dict]:
        """Recupera lo storico editoriale"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Post)-[:COVERS]->(t:Topic)
                RETURN p.title as title, 
                       p.created_at as created_at,
                       collect(t.name) as topics
                ORDER BY p.created_at DESC
                """
            )
            return [dict(record) for record in result]
    
    def query(self, cypher: str, params: Dict = None) -> List[Dict]:
        """Esecuzione query generica"""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]
    
    def close(self):
        self.driver.close()
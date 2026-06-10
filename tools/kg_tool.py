# tools/kg_tool.py
from kg.neo4j_manager import Neo4jManager
from langchain.tools import tool
from typing import List, Dict, Any

class KGQueryTool:
    """
    Knowledge Graph tool per querying e updating structured knowledge
    """
    def __init__(self, kg_manager: Neo4jManager):
        self.kg = kg_manager
    
    def query_topics(self, topic_filter: str = None) -> List[Dict]:
        """Query topics dal KG"""
        if topic_filter:
            cypher = """
                MATCH (t:Topic)
                WHERE t.name CONTAINS $filter
                RETURN t.name as name, t.description as description
                LIMIT 20
            """
            params = {"filter": topic_filter}
        else:
            cypher = """
                MATCH (t:Topic)
                RETURN t.name as name, t.description as description
                LIMIT 50
            """
            params = {}
        
        return self.kg.query(cypher, params)
    
    def get_related(self, topic: str, depth: int = 1) -> List[str]:
        """Trova topic correlati"""
        cypher = f"""
            MATCH (t:Topic {{name: $topic}})-[:RELATED_TO*1..{depth}]-(related:Topic)
            WHERE t.name <> related.name
            RETURN DISTINCT related.name as topic
            LIMIT 10
        """
        results = self.kg.query(cypher, {"topic": topic})
        return [r['topic'] for r in results]
    
    def get_post_history(self, limit: int = 10) -> List[Dict]:
        """Recupera storico post per evitare ridondanza"""
        cypher = """
            MATCH (p:Post)-[:COVERS]->(t:Topic)
            RETURN p.title as title, p.created_at as date, collect(t.name) as topics
            ORDER BY p.created_at DESC
            LIMIT $limit
        """
        return self.kg.query(cypher, {"limit": limit})
    
    def update_post(self, title: str, content: str, topics: List[str], sources: List[str]) -> str:
        """Aggiorna KG con nuovo post (wrapper per add_post)"""
        return self.kg.add_post(title, content, topics, sources)

kg_tool_instance = None

def get_kg_tool(kg_manager):
    global kg_tool_instance
    if kg_tool_instance is None:
        kg_tool_instance = KGQueryTool(kg_manager)
    return kg_tool_instance


@tool
def kg_search(topic: str) -> str:
    """Search the Knowledge Graph for topics related to the given topic (or multiple topics separated by commas, e.g., 'Golf, Technique, Improvement').
    Returns related topics and their connections. Use this tool to discover
    broader context, related subjects, and previously covered topics."""
    global kg_tool_instance
    if kg_tool_instance is None:
        return "Knowledge Graph not available."
    try:
        # Dividi per virgola se vengono passati più topic contemporaneamente
        topics = [t.strip() for t in topic.split(',') if t.strip()]
        all_related = []
        for t in topics:
            related = kg_tool_instance.get_related(t, depth=1)
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
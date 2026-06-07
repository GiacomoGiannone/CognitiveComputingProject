# app.py (parte principale)
import os
from dotenv import load_dotenv
from kg.neo4j_manager import Neo4jManager
from graph.workflow import create_blog_workflow

load_dotenv(override=True)

def main():
    # Verifica Tavily
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ ERROR: TAVILY_API_KEY not set in .env file")
        return
    
    # Neo4j opzionale
    try:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        kg = Neo4jManager(
            neo4j_uri,
            neo4j_user,
            neo4j_password
        )
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"⚠️ Neo4j not available: {e}")
        kg = None
    
    # Crea workflow
    app = create_blog_workflow(kg)
    
    # Stato iniziale - NOTA: current_topic sarà impostato dal planner
    initial_state = {
        "blog_domain": "Sport",
        "kg_manager": kg,
        "current_topic": None,  # Sarà impostato dal planner
        "extracted_graph_topics": [],  # Topic generici estratti dal planner
        "max_post_length": 1500,
        "iteration": 0,
        "max_iterations": 2,
        "requires_regeneration": False,
        "requires_research": False,
        "editorial_plan": "",
        "research_results": {},
        "draft_post": {},
        "fact_check_passed": False,
        "fact_check_results": {},
        "review_action": "",
        "modification_feedback": "",
        "final_post": {}
    }
    
    # Esegui workflow
    print("\n🚀 Starting workflow...\n")
    
    for output in app.stream(initial_state):
        if output:
            print(f"Step output: {list(output.keys())}")
        
        # Controlla se 'final_post' è presente in uno dei nodi eseguiti nello step
        if output and any('final_post' in node_data for node_data in output.values() if isinstance(node_data, dict)):
            print("\n✅ Post published!")
            break
    
    if kg:
        kg.close()

if __name__ == "__main__":
    main()
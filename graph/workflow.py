# workflow.py
from langgraph.graph import END, StateGraph
from graph.state import BlogState

# Importa le funzioni, non i moduli
from agents.planner_agent import planner_agent
from agents.research_agent import research_agent
from agents.writer_agent import writer_agent
from agents.fact_checker import fact_check_agent
from agents.human_review import human_review_agent
from agents.score_agent import score_agent

from kg.neo4j_manager import Neo4jManager

def create_blog_workflow(kg_manager: Neo4jManager):
    workflow = StateGraph(BlogState)
    
    # Aggiungi nodi con le funzioni
    workflow.add_node("planner", planner_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("fact_check", fact_check_agent)
    workflow.add_node("score", score_agent)
    workflow.add_node("human_review", human_review_agent)
    
    # Definisci edge
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "research")
    workflow.add_edge("research", "writer")
    workflow.add_edge("writer", "fact_check")
    workflow.add_edge("writer", "score")
    
    # Conditional edges
    def after_fact_check(state):
        if state.get('fact_check_passed', False):
            return "human_review"
        else:
            return "writer"
    
    workflow.add_conditional_edges(
        "fact_check",
        after_fact_check,
        {
            "human_review": "human_review",
            "writer": "writer"
        }
    )
    
    def after_review(state):
        action = state.get('review_action', '')
        
        if action == 'approved':
            if state.get('proceed_to_next_topic', False):
                return "planner"
            else:
                return END
        elif action == 'modify_requested':
            return "writer"
        elif action == 'rejected' or state.get('requires_research'):
            if state.get('iteration', 0) < state.get('max_iterations', 3):
                return "research"
            else:
                return END
        else:
            return END
    
    workflow.add_conditional_edges(
        "human_review",
        after_review,
        {
            "writer": "writer",
            "research": "research",
            "planner": "planner",
            END: END
        }
    )
    
    return workflow.compile()
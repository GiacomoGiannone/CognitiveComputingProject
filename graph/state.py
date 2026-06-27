from typing import TypedDict, List, Dict, Any


class BlogState(TypedDict, total=False):

    # Core inputs
    blog_domain: str
    current_topic: str
    all_topics: List[str]
    editorial_plan: str
    extracted_graph_topics: List[str]

    # Research + RAG
    search_results: List[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]
    research_results: Dict[str, Any]

    # KG context
    kg_manager: Any
    kg_context: Dict[str, Any]

    # Writing + fact check
    draft_post: Dict[str, Any]
    extracted_claims: List[str]
    verified_claims: List[Dict[str, Any]]
    fact_check_passed: bool
    fact_check_results: Dict[str, Any]

    # Evaluation
    quality_evaluation: Dict[str, Any]
    quality_passed: bool
    barely_passed: bool

    # Human review + control flow
    review_action: str
    modification_feedback: str
    final_post: Dict[str, Any]
    requires_regeneration: bool
    requires_research: bool
    iteration: int
    max_iterations: int
    max_post_length: int

    # Optional debug/trace
    thoughts: List[str]
    next_agent: str
from typing import TypedDict, List, Dict, Any


class BlogState(TypedDict, total=False):

    # Core inputs
    blog_domain: str
    current_topic: str
    extracted_graph_topics: List[str]

    # Research + RAG
    research_results: Dict[str, Any]

    # KG context
    kg_manager: Any

    # Writing + fact check
    draft_post: Dict[str, Any]
    fact_check_passed: bool
    fact_check_results: Dict[str, Any]

    # Evaluation
    quality_passed: bool
    barely_passed: bool

    # Human review + control flow
    review_action: str
    modification_feedback: str
    final_post: Dict[str, Any]
    requires_research: bool
    iteration: int
    max_iterations: int
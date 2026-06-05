from typing import TypedDict, List, Dict, Any


class BlogState(TypedDict):

    blog_domain: str

    topic: str

    editorial_plan: list

    search_results: list

    retrieved_docs: list

    kg_context: dict

    extracted_claims: list

    verified_claims: list

    draft_post: str

    review_action: str

    thoughts: list[str]

    next_agent: str
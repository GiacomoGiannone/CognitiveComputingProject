from langgraph.graph import StateGraph
from langgraph.graph import END

from agents.state import AgentState
from nodes.kg_update_node import kg_update_node
from nodes.planner_node import planner_node
from nodes.research_node import research_node
from nodes.review_node import review_node
from nodes.writer_node import writer_node

builder = StateGraph(AgentState)

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "research",
    research_node
)

builder.add_node(
    "writer",
    writer_node
)

builder.add_node(
    "review",
    review_node
)

builder.add_node(
    "kg_update",
    kg_update_node
)

builder.set_entry_point("planner")

builder.add_edge("planner", "research")

builder.add_edge("research", "writer")

builder.add_edge("writer", "review")

builder.add_edge("review", "kg_update")

builder.add_edge("kg_update", END)

#per il ciclo ReAct dovremmo aggiungere un edge che vada all'indietro

graph = builder.compile()
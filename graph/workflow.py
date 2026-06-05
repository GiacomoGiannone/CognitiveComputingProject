from langgraph.graph import StateGraph

from state import BlogState
from agents import planner_agent

builder = StateGraph(BlogState)

builder.add_node(
    "planner",
    planner_agent
)

builder.set_entry_point(
    "planner"
)

builder.add_edge(
    "planner",
    "planner"
)

graph = builder.compile()
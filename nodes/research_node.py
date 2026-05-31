from agents.research_agent import ResearchAgent

def research_node(state):
    print("[research] start")
    topic = state["chosen_topic"]
    revision_query = state.get("revision_query")
    if revision_query:
        topic = f"{topic} {revision_query}"
        state["revision_query"] = ""
    print(f"[research] query={topic}")
    agent = ResearchAgent(name="research", state=state)
    outputs = agent.perform_research(topic)

    state.update(outputs)
    sources_count = len(outputs.get("tool_outputs", {}).get("search", {}).get("results", []))
    print(f"[research] sources={sources_count}")
    return state
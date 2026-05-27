from agents.research_agent import ResearchAgent

def research_node(state):
    print("[research] start")
    print(f"[research] state={state}")
    topic = state["chosen_topic"]
    print(f"[research] query={topic}")
    agent = ResearchAgent(name="research", state=state)
    outputs = agent.perform_research(topic)

    state.update(outputs)
    sources_count = len(outputs.get("tool_outputs", {}).get("search", {}).get("results", []))
    print(f"[research] sources={sources_count}")
    print(f"[research] state_end={state}")
    return state
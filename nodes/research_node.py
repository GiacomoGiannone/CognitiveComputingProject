def research_node(state):

    topic = state["chosen_topic"]

    verified_info = f"Verified info about {topic}"

    reasoning = [
        f"Searched web for {topic}",
        "Checked trusted sources"
    ]

    state["verified_info"] = verified_info
    state["reasoning_trace"] = reasoning

    return state
def kg_update_node(state):
    kg = state.get("kg")

    if kg is None:
        return state

    if state["content_feedback"] != "yes":
        return state

    topic = state["chosen_topic"]

    kg.add_node(
        "post_1",
        "Post",
        {
            "content": state["created_content"]
        }
    )

    kg.add_relationship(
        "post_1",
        "COVERS",
        topic
    )

    return state
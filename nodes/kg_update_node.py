def kg_update_node(state):
    print(f"[kg_update] state={state}")
    kg = state.get("kg")

    if kg is None:
        return state

    if state["content_feedback"] != "yes":
        return state

    topic_name = state["chosen_topic"]

    # Find or create the Topic node so the relationship target exists.
    existing_topics = kg.get_nodes_by_type("Topic")
    topic_id = None
    for node_id, node_data in existing_topics.items():
        name = node_data.get("properties", {}).get("name", "")
        if name.strip().lower() == str(topic_name).strip().lower():
            topic_id = node_id
            break

    if topic_id is None:
        topic_id = f"topic_{len(kg.data['nodes']) + 1}"
        kg.add_node(
            topic_id,
            "Topic",
            {
                "name": topic_name
            }
        )

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
        topic_id
    )

    print(f"[kg_update] state_end={state}")

    return state
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

    # Generate a unique post ID instead of overwriting post_1.
    existing_posts = kg.get_nodes_by_type("Post")
    max_id = 0
    for node_id in existing_posts.keys():
        if node_id.startswith("post_"):
            suffix = node_id.split("post_")[-1]
            if suffix.isdigit():
                max_id = max(max_id, int(suffix))
    new_post_id = f"post_{max_id + 1}"

    kg.add_node(
        new_post_id,
        "Post",
        {
            "content": state["created_content"]
        }
    )

    kg.add_relationship(
        new_post_id,
        "COVERS",
        topic_id
    )

    print(f"[kg_update] state_end={state}")

    return state
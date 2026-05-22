def review_node(state):

    feedback = input(
        "Approve post? (yes/no): "
    )

    state["content_feedback"] = feedback

    return state
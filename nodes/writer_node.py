def writer_node(state):

    info = state["verified_info"]

    draft = f"""
    Blog post draft:

    {info}
    """

    state["created_content"] = draft

    return state
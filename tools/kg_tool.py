def get_related_topics(topic):

    query = """
    MATCH (t:Topic {name:$topic})
    -[:RELATED_TO]->
    (other:Topic)

    RETURN other.name as topic
    """

    return kg.query(
        query,
        {"topic": topic}
    )
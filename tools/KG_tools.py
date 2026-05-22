from kg.KG import KG

kg = KG()

def get_topics_already_covered(topic_id):
    posts = kg.get_posts_about_topic(topic_id)

    return len(posts) > 0

def get_N_most_recent_topics( N=5):
    #first we get the posts
    posts = kg.get_nodes_by_type("Post")

    #now we sort them by date
    sorted_posts = sorted(posts.items(), key=lambda x: x[1]["properties"].get("date", "1970-01-01"), reverse=True)

    topics = [] #this will be the output list of topics
    seen_topic_ids = set()

    for post_id, post_data in sorted_posts:
        if len(topics) >= N:
            break
        
        topic_ids = kg.get_connected_nodes(post_id, relation="COVERS")
        for topic_id in topic_ids:
            if len(topics) >= N:
                break
            if topic_id in seen_topic_ids:
                continue

            topic_data = kg.get_node(topic_id)
            if not topic_data or topic_data.get("type") != "Topic":
                continue

            topics.append({"id": topic_id, **topic_data})
            seen_topic_ids.add(topic_id)

    return topics


import google.genai as genai

from kg.KG import KG


def main():
	kg = KG(path="kg.json")

	kg.add_node(
		node_id="topic_ai",
		node_type="Topic",
		properties={"name": "Artificial Intelligence"}
	)
	kg.add_node(
		node_id="post_llm_intro",
		node_type="Post",
		properties={"title": "Intro to LLMs", "url": "https://example.com/llm"}
	)

	kg.add_relationship(
		source="post_llm_intro",
		relation="COVERS",
		target="topic_ai"
	)

	topic = kg.get_node("topic_ai")
	posts = kg.get_posts_about_topic("topic_ai")

	print("Topic:", topic)
	print("Posts about topic:", posts)


if __name__ == "__main__":
	main()


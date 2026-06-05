#use ollama llama3.1
import ollama

llm = ollama.Chat(model="llama3.1")


def planner_agent(state):

    prompt = f"""
    Domain: {state['blog_domain']}

    Genera 5 topic futuri
    evitando ridondanza.
    """

    response = llm.invoke(prompt)

    return {
        "editorial_plan": response.content
    }
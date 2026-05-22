from tools.KG_tools import get_N_most_recent_topics, get_topics_already_covered
from agents.state import AgentState
import ollama

class Planner_agent:
    def __init__(self, state, model, model_name="llama3.1"):
        self.state = state
        self.model = model
        self.model_name = model_name

    def _generate(self, prompt):
        if hasattr(self.model, "chat"):
            response = self.model.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]

        if hasattr(self.model, "generate"):
            response = self.model.generate(prompt)
            return response.text if hasattr(response, "text") else str(response)

        if isinstance(self.model, str):
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]

        raise TypeError("model must be an Ollama Client, a model name string, or expose generate().")

    def choose_topic(self):
        #Logica per scegliere un topic da approfondire, basata su user_input e recent_topics
        
        #prima dobbiamo fare la query al KG per ottenere i topic recenti
        recent_topics = get_N_most_recent_topics(N=5)

        user_input = self.state.get("user_input", "")
        user_input_norm = user_input.strip().lower()
        covered_match = None
        for topic in recent_topics:
            topic_name = topic.get("properties", {}).get("name", "")
            if not topic_name:
                continue
            if topic_name.lower() in user_input_norm:
                if get_topics_already_covered(topic["id"]):
                    covered_match = topic_name
                    break

        #ora possiamo costruire il prompt per il modello, includendo l'input dell'utente e i topic recenti
        prompt = f"User input: {user_input}\n"
        prompt += "Recent topics in the KG:\n"
        for topic in recent_topics:
            topic_name = topic.get("properties", {}).get("name", "unknown")
            prompt += f"- {topic_name} (id: {topic['id']})\n"
        if covered_match:
            prompt += (
                f"The user asked about a topic already covered: {covered_match}. "
                "Reply in Italian with one sentence that says it was already covered and optionally suggest a different topic."
            )
        else:
            prompt += "Based on the user input and the recent topics, choose one topic to research further. If none of the recent topics are relevant, you can suggest a new topic."

        #Ora chiamiamo il modello per ottenere la scelta del topic
        response_text = self._generate(prompt)
        #Supponiamo che il modello risponda con l'id del topic scelto o con un nuovo topic da aggiungere
        chosen_topic = response_text.strip()
        self.state['recent_topics'] = recent_topics
        self.state['chosen_topic'] = chosen_topic
        print("Planner state:", self.state)
        return chosen_topic

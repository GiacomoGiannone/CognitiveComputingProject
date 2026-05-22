from tools.KG_tools import get_N_most_recent_topics, get_topics_already_covered
from agents.state import AgentState
import ollama

class Planner_agent:
    def __init__(self, state, model, kg, model_name="llama3.1"):
        self.state = state
        self.model = model
        self.kg = kg
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
        recent_topics = get_N_most_recent_topics(self.kg, N=5)
        uncovered_topics = []

        user_input = self.state.get("user_input", "")
        user_input_norm = user_input.strip().lower()
        covered_match = None
        covered_topics = []
        for topic in recent_topics:
            topic_name = topic.get("properties", {}).get("name", "")
            if not topic_name:
                continue
            if get_topics_already_covered(self.kg, topic["id"]):
                covered_topics.append(topic_name)
            else:
                uncovered_topics.append(topic)
            if topic_name.lower() in user_input_norm:
                if get_topics_already_covered(self.kg, topic["id"]):
                    covered_match = topic_name
                    break

        #ora possiamo costruire il prompt per il modello, includendo l'input dell'utente e i topic recenti
        prompt = f"User input: {user_input}\n"
        prompt += "Candidate topics (not covered yet):\n"
        for topic in uncovered_topics:
            topic_name = topic.get("properties", {}).get("name", "unknown")
            prompt += f"- {topic_name} (id: {topic['id']})\n"
        if covered_topics:
            covered_list = ", ".join(covered_topics)
            prompt += (
                f"Already covered topics in the KG: {covered_list}. "
                "Start your reply by stating that these topics were already covered."
            )
        if covered_match:
            prompt += (
                f"The user asked about a topic already covered: {covered_match}. "
                "You must NOT choose a covered topic. Reply in Italian with one sentence that says it was already covered and then suggest a different topic."
            )
        else:
            prompt += "Based on the user input and the candidate topics, choose one topic to research further. If none are relevant, suggest a new topic."

        #Ora chiamiamo il modello per ottenere la scelta del topic
        response_text = self._generate(prompt)
        #Supponiamo che il modello risponda con l'id del topic scelto o con un nuovo topic da aggiungere
        chosen_topic = response_text.strip()
        self.state['recent_topics'] = recent_topics
        self.state['chosen_topic'] = chosen_topic
        print("Planner state:", self.state)
        return chosen_topic

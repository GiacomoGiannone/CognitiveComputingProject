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

        prompt_reasoning = f"User input: {user_input}\n"
        prompt_reasoning += "Candidate topics (not covered yet):\n"
        for topic in uncovered_topics:
            topic_name = topic.get("properties", {}).get("name", "unknown")
            prompt_reasoning += f"- {topic_name} (id: {topic['id']})\n"
        if covered_topics:
            covered_list = ", ".join(covered_topics)
            prompt_reasoning += (
                f"Already covered topics in the KG: {covered_list}. "
                "You must NOT choose any of these covered topics. "
            )
        if covered_match:
            prompt_reasoning += (
                f"The user specifically asked about a topic already covered: {covered_match}. "
                "You must NOT choose this topic. "
            )
        
        prompt_reasoning += (
            "Analyze the user input to thoroughly understand their intent. "
            "Based on the user input, choose EXACTLY ONE specific topic to research further.\n"
            "If the user asks for a specific subject like 'calciomercato' (transfer market), make sure your chosen topic reflects that "
            "(for example, choosing a specific team and adding 'calciomercato', or focusing generally on 'calciomercato Serie A').\n"
            "Explain your reasoning for choosing this topic in Italian. Ensure the chosen topic perfectly matches what the user wants to write about."
        )

        # Prima chiamata: Chiediamo al modello di spiegare il ragionamento
        reasoning = self._generate(prompt_reasoning).strip()
        
        # Seconda chiamata: Estraiamo SOLO la stringa del topic basandoci sul ragionamento
        prompt_extraction = (
            f"Here is an explanation of a chosen topic:\n\"{reasoning}\"\n\n"
            "Based on this reasoning, construct a CONCISE, highly effective search query (2-5 words) to find recent news for this specific context.\n"
            "If the context is about a team's transfer market, output e.g., 'Milan calciomercato' or 'Juventus calciomercato'.\n"
            "If it's general transfer market, output 'Calciomercato Serie A'.\n"
            "If the user just wants team news, output e.g., 'Napoli ultime notizie'.\n\n"
            "Reply ONLY with the exact search query and nothing else. "
            "Do not include any punctuation, quotes, or explanations."
        )
        
        chosen_topic = self._generate(prompt_extraction).strip()

        self.state['recent_topics'] = recent_topics
        self.state['chosen_topic'] = chosen_topic
        
        if 'reasoning_trace' not in self.state:
            self.state['reasoning_trace'] = []
        self.state['reasoning_trace'].append(f"Planner reasoning: {reasoning}")
        
        print(f"\n[planner] reasoning: {reasoning}\n")
        print("Planner state:", self.state)
        return chosen_topic

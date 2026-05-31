
import ollama

class WriterAgent:
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

    def write_post(self):
        topic = self.state.get("chosen_topic", "Argomento sconosciuto")
        tool_outputs = self.state.get("tool_outputs", {})
        clean_docs = tool_outputs.get("clean_docs", [])

        # Estrai il testo utile per il RAG
        context_text = ""
        for i, doc in enumerate(clean_docs):
            if doc.get("text"):
                # limitiamo la lunghezza di ogni documento per non sfondare il context window
                doc_text = doc["text"][:2000] 
                context_text += f"\n--- Fonte {i+1}: {doc.get('title', 'Senza Titolo')} ---\n{doc_text}\n"

        feedback_detail = self.state.get("content_feedback_detail", "").strip()
        feedback_section = ""
        if feedback_detail:
            feedback_section = (
                "\nNOTE DI REVISIONE (da rispettare):\n"
                f"- {feedback_detail}\n"
            )

        prompt = (
            f"Sei un giornalista sportivo esperto. Scrivi un articolo di blog avvincente in italiano basato sul seguente argomento: {topic}.\n\n"
            f"UTILIZZA ESCLUSIVAMENTE le seguenti informazioni estratte dal web (RAG) per scrivere l'articolo. "
            "Se non ci sono informazioni sufficienti, scrivi un articolo breve evidenziando solo ciò che sai dalle fonti.\n\n"
            f"FONTI:\n{context_text}\n\n"
            f"{feedback_section}"
            "REGOLE:\n"
            "- Titolo accattivante.\n"
            "- Formattazione in Markdown (usa grassetti, liste puntate se necessario).\n"
            "- Non inventare notizie che non sono presenti nelle fonti.\n"
            "- Includi riferimenti o cita le fonti menzionate.\n"
        )

        draft = self._generate(prompt)
        self.state["created_content"] = draft
        return draft
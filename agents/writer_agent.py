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

    def _format_sources(self, clean_docs):
        """Formatta le fonti con snippet di testo invece di riferimenti inventati"""
        sources_text = "\n\n---\n## 📚 Fonti e approfondimenti\n\n"
        
        for i, doc in enumerate(clean_docs[:5]):  # Max 5 fonti
            text = doc.get("text", "")
            title = doc.get("title", "Senza titolo")
            url = doc.get("url", "URL non disponibile")
            source_type = doc.get("source_type", "web")
            
            # Prendi uno snippet significativo (prime 200-300 caratteri)
            snippet = text[:300].strip()
            if len(text) > 300:
                snippet += "..."
            
            sources_text += f"**Fonte {i+1}:** {title}\n"
            sources_text += f"- **URL:** {url}\n"
            sources_text += f"- **Estratto:** \"{snippet}\"\n\n"
        
        return sources_text

    def write_post(self):
        topic = self.state.get("chosen_topic", "Argomento sconosciuto")
        tool_outputs = self.state.get("tool_outputs", {})
        clean_docs = tool_outputs.get("clean_docs", [])
        
        # Se non ci sono documenti, usa un messaggio di default
        if not clean_docs:
            return f"Non ho trovato sufficienti informazioni su {topic} per scrivere un articolo."

        # Estrai il testo utile per il contesto (massimo 1500 caratteri per documento)
        context_text = ""
        for i, doc in enumerate(clean_docs[:5]):  # Limita a 5 fonti
            if doc.get("text"):
                doc_text = doc["text"][:1500]
                context_text += f"\n--- Fonte {i+1}: {doc.get('title', 'Senza Titolo')} ---\n{doc_text}\n"

        feedback_detail = self.state.get("content_feedback_detail", "").strip()
        feedback_section = ""
        if feedback_detail:
            feedback_section = (
                "\nNOTE DI REVISIONE (da rispettare):\n"
                f"- {feedback_detail}\n"
            )

        manual_facts = self.state.get("manual_facts", "").strip()
        manual_section = ""
        if manual_facts:
            manual_section = (
                "\nINFORMAZIONI CORRETTE/INTEGRATE (da usare):\n"
                f"- {manual_facts}\n"
            )

        # Prompt che chiede ESPLICITAMENTE di NON includere le fonti
        prompt = (
            f"Sei un giornalista sportivo esperto. Scrivi un articolo di blog avvincente in italiano basato sul seguente argomento: {topic}.\n\n"
            f"UTILIZZA ESCLUSIVAMENTE le seguenti informazioni estratte dal web per scrivere l'articolo.\n\n"
            f"CONTENUTO DELLE FONTI:\n{context_text}\n\n"
            f"{feedback_section}"
            f"{manual_section}"
            "REGOLE IMPORTANTI:\n"
            "- Scrivi SOLO l'articolo, niente altro.\n"
            "- NON includere sezioni di fonti, bibliografia, riferimenti o note a piè di pagina.\n"
            "- NON scrivere 'Fonti:', 'Riferimenti:', o sezioni simili.\n"
            "- NON usare markdown per link o citazioni.\n"
            "- Scrivi in modo naturale, come un articolo di blog.\n"
            "- Usa titoli e sottotitoli in markdown (es. ## Sottotitolo).\n"
            "- Non inventare informazioni non presenti nelle fonti.\n"
            "- Alla fine dell'articolo, fermati. Non aggiungere altro.\n"
        )

        draft = self._generate(prompt)
        
        # Pulisci eventuali sezioni di fonti che il modello ha aggiunto comunque
        import re
        # Rimuovi sezioni che iniziano con "Fonti", "Riferimenti", "Sources", etc.
        patterns = [
            r'\n---+\s*\n+\s*📚?\s*Fonti.*$',
            r'\n---+\s*\n+\s*Riferimenti.*$',
            r'\n---+\s*\n+\s*Sources.*$',
            r'\n##\s*📚?\s*Fonti.*$',
            r'\n##\s*Riferimenti.*$',
            r'\n\[FONTI\].*$',
            r'\n\*\*Fonti\*\*:.*$',
        ]
        
        for pattern in patterns:
            draft = re.sub(pattern, '', draft, flags=re.IGNORECASE | re.DOTALL)
        
        # Aggiungi le fonti SOLO DOPO, in modo pulito e senza duplicati
        sources_section = self._format_sources(clean_docs[:5])
        final_article = draft.strip() + sources_section
        
        self.state["created_content"] = final_article
        return final_article
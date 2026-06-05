# planner_agent.py
import ollama
from kg.neo4j_manager import Neo4jManager
from typing import List, Dict

class PlannerAgent:
    def __init__(self, kg_manager: Neo4jManager, llm_model: str = "llama3.1"):
        self.kg = kg_manager
        self.llm_model = llm_model

    def _chat(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    
    def get_covered_topics_context(self) -> str:
        """Recupera i topic già coperti dal KG"""
        covered = self.kg.get_covered_topics()
        if not covered:
            return "Nessun topic è stato ancora coperto."
        
        history = self.kg.get_editorial_history()
        
        context = "Topic già coperti:\n"
        for topic in covered:
            context += f"- {topic}\n"
        
        context += "\nStorico recente:\n"
        for post in history[:5]:  # ultimi 5 post
            context += f"- {post['title']} (topic: {', '.join(post['topics'])})\n"
        
        return context
    
    def get_topic_suggestions(self, domain: str, num_suggestions: int = 5) -> List[Dict]:
        """Genera suggerimenti di topic basati sul grafo"""
        
        covered_topics = set(self.kg.get_covered_topics())
        
        # Recupera topic correlati da KG
        related_suggestions = set()
        for topic in covered_topics:
            related = self.kg.get_related_topics(topic, limit=3)
            related_suggestions.update(related)
        
        # Filtra già coperti
        new_suggestions = [t for t in related_suggestions if t not in covered_topics]
        
        prompt = f"""
        Domain: {domain}
        
        {self.get_covered_topics_context()}
        
        Task: Genera {num_suggestions} topic per nuovi post.
        Evita ridondanza con i topic già coperti.
        
        I topic suggeriti dal grafo correlato sono: {new_suggestions if new_suggestions else 'nessuno'}
        
        Per ogni topic, fornisci:
        1. Nome del topic
        2. Breve descrizione
        3. Perché è interessante per i lettori
        4. Quale gap editoriale colma
        
        Rispondi in formato JSON.
        """
        
        response = self._chat(prompt)
        
        # Parsing della risposta (semplificato)
        suggestions = self._parse_suggestions(response)
        
        return suggestions
    
    def _parse_suggestions(self, response_text: str) -> List[Dict]:
        """Parsa la risposta dell'LLM in formato strutturato"""
        # Implementazione semplificata - in produzione usare JSON parser
        suggestions = [
            {
                "topic": "Tecniche avanzate di virata",
                "description": "Approfondimento sulle tecniche di virata in condizioni difficili",
                "rationale": "Topic molto richiesto ma ancora non coperto"
            }
        ]
        return suggestions
    
    def generate_editorial_plan(self, domain: str, horizon_days: int = 30) -> Dict:
        """Genera piano editoriale completo"""
        
        suggestions = self.get_topic_suggestions(domain)
        
        prompt = f"""
        Domain: {domain}
        
        Piano editoriale per i prossimi {horizon_days} giorni.
        
        Topic suggeriti: {suggestions}
        
        Genera un piano sequenziale che:
        1. Giustifichi l'ordine dei post
        2. Assicuri diversità e copertura del dominio
        3. Colleghi i nuovi post con quelli esistenti
        4. Specifichi per ogni post: titolo, topic, tipo (how-to/review/news), lunghezza stimata
        
        Restituisci il piano in formato strutturato.
        """
        
        response = self._chat(prompt)
        
        return {
            "plan": response,
            "suggestions": suggestions,
            "horizon_days": horizon_days
        }


# planner_agent_simple.py (versione base per integrazione con il tuo file)
def planner_agent(state):
    """Versione semplificata per integrazione con il workflow"""
    kg = state.get('kg_manager')
    
    if kg:
        # Usa il planner avanzato
        planner = PlannerAgent(kg, state.get("llm_model", "llama3.1"))
        plan = planner.generate_editorial_plan(state['blog_domain'])
        return {"editorial_plan": plan}
    else:
        # Fallback base
        prompt = f"""
        Domain: {state['blog_domain']}
        Genera 5 topic futuri evitando ridondanza.
        """
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"editorial_plan": response["message"]["content"]}
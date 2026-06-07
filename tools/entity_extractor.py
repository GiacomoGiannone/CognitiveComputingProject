# tools/entity_extractor.py
"""
Entity Extractor - Estrae topic generici dal titolo di un articolo
Questo tool risolve il problema architetturale di avere topic troppo specifici nel KG.
Converte un titolo lungo e specifico in 2-3 topic generici e riutilizzabili.
"""

import ollama
import json
import re
from typing import List, Dict, Any


class EntityExtractor:
    def __init__(self, llm_model: str = "llama3.1"):
        self.llm_model = llm_model
    
    def _call_llm(self, prompt: str) -> str:
        """Chiamata a Ollama"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ LLM Error in EntityExtractor: {e}")
            return ""
    
    def extract_graph_topics(self, post_title: str, blog_domain: str = None) -> List[str]:
        """
        Estrae 2-3 topic generici dal titolo specifico dell'articolo.
        
        Esempio:
            Input:  "La gestione dell'ansia prima di un grande evento sportivo"
            Output: ["Ansia", "Psicologia Sportiva", "Performance Mentale"]
        
        Args:
            post_title: Il titolo specifico del post
            blog_domain: Il dominio del blog (opzionale, per aiutare il contesto)
        
        Returns:
            Lista di 2-3 topic generici estratti
        """
        
        print(f"\n🔍 Entity Extraction: Extracting topics from '{post_title[:50]}...'")
        
        domain_context = f"The blog domain is: {blog_domain}" if blog_domain else ""
        
        prompt = f"""
        You are an expert at extracting general, reusable topics from specific article titles.

        TASK: Given a specific article title, extract 2-3 generic, reusable topics (or tags) that:
        - Are broader and more general than the article title
        - Can be used to find related articles in a knowledge graph
        - Are single words or short phrases (max 3 words each)
        - Are meaningful and categorizable
        - Are the most obvious and the more general possible

        EXAMPLE:
        - Title: "La gestione dell'ansia prima di un grande evento sportivo"
        - Topics: ["Ansia", "Psicologia Sportiva", "Performance Mentale"]

        EXAMPLE 2:
        - Title: "Come scegliere il primo vela per principianti"
        - Topics: ["Vela", "Nautica per Principianti"]

        EXAMPLE 3:
        - Title: "Tecniche avanzate di nodo per appassionati di arrampicata"
        - Topics: ["Nodi", "Arrampicata", "Tecniche di Sicurezza"]

        {domain_context}

        ARTICLE TITLE: "{post_title}"

        OUTPUT FORMAT: Return a JSON object with this exact structure:
        {{
        "topics": ["topic1", "topic2", "topic3"]
        }}

        Return ONLY the JSON object, no other text.
        """
        
        response = self._call_llm(prompt)
        
        try:
            # Prova a trovare JSON nella risposta
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                topics = data.get('topics', [])
                
                # Valida che abbiamo topic validi
                if isinstance(topics, list) and len(topics) > 0:
                    # Limita a max 3 topic e pulisci
                    topics = [t.strip() for t in topics[:3] if t.strip()]
                    print(f"✅ Extracted generic topics: {topics}")
                    return topics
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON: {e}")
        except Exception as e:
            print(f"⚠️ Extraction error: {e}")
        
        # Fallback: estrai da ricerca semplice
        print(f"⚠️ Using fallback extraction for: {post_title[:30]}...")
        return self._fallback_extraction(post_title)
    
    def _fallback_extraction(self, post_title: str) -> List[str]:
        """
        Estrazione di fallback quando l'LLM non funziona.
        Usa semplice keyword extraction.
        """
        
        # Rimuovi stopwords comuni e articoli
        stopwords = {
            'la', 'le', 'il', 'i', 'lo', 'gli', 'un', 'una', 
            'da', 'per', 'con', 'di', 'a', 'o', 'e', 'è', 'come', 'nel', 'sulla',
            'durante', 'prima', 'dopo'
        }
        
        # Dividi in parole e filtra
        words = [w.strip('.,!?;:') for w in post_title.lower().split()]
        significant_words = [
            w for w in words 
            if len(w) > 3 and w not in stopwords and w.isalpha()
        ]
        
        # Ritorna le 2-3 prime parole significative
        topics = significant_words[:3]
        if not topics:
            # Fallback estremo
            topics = [word for word in words[:3] if len(word) > 2]
        
        print(f"📌 Fallback topics: {topics}")
        return topics if topics else ["General"]
    
    def extract_and_validate(self, post_title: str, blog_domain: str = None) -> Dict[str, Any]:
        """
        Estrae topic e ritorna un dict con validazione.
        Utile per uso nei tools.
        
        Returns:
            {
                "success": bool,
                "topics": List[str],
                "post_title": str,
                "message": str
            }
        """
        
        try:
            topics = self.extract_graph_topics(post_title, blog_domain)
            return {
                "success": True,
                "topics": topics,
                "post_title": post_title,
                "message": f"Successfully extracted {len(topics)} topics"
            }
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return {
                "success": False,
                "topics": [],
                "post_title": post_title,
                "message": f"Error: {str(e)}"
            }


# Instanza globale (lazy loading)
_extractor = None

def get_entity_extractor(llm_model: str = "llama3.1") -> EntityExtractor:
    """Factory function per ottenere l'estrattore"""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor(llm_model)
    return _extractor


def extract_topics_from_title(post_title: str, blog_domain: str = None) -> List[str]:
    """
    Funzione helper semplice per estrarre topic.
    Uso: from tools.entity_extractor import extract_topics_from_title
    """
    extractor = get_entity_extractor()
    return extractor.extract_graph_topics(post_title, blog_domain)

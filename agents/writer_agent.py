# writer_agent.py
import ollama
from typing import Dict

class WriterAgent:
    def __init__(self, llm_model: str = "llama3.1"):
        self.llm_model = llm_model

    def _chat(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    
    def write_post(self, topic: str, research_results: Dict, max_length: int = 1500) -> str:
        """Scrive un post basato sulla ricerca"""
        
        sources = research_results.get('sources', [])
        sources_text = "\n".join([
            f"- {s['title']} ({s['url']})" for s in sources[:3]
        ])
        
        prompt = f"""
        Write a blog post about: {topic}
        
        Research findings:
        {research_results.get('research_summary', 'No research available')}
        
        Sources to cite:
        {sources_text}
        
        Requirements:
        - Max length: {max_length} words
        - Include citations inline [Source: URL]
        - Engaging and informative tone
        - Clear structure with headings
        - Target audience: enthusiasts and practitioners
        
        Write the complete blog post with title and body.
        """
        
        response = self._chat(prompt)
        
        # Aggiungi metadata
        post = {
            'title': self._extract_title(response),
            'content': response,
            'sources': sources,
            'topic': topic,
            'word_count': len(response.split())
        }
        
        return post
    
    def _extract_title(self, content: str) -> str:
        """Estrae il titolo dal contenuto"""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('#') or (len(line) < 100 and len(line) > 10):
                return line.strip('# ').strip()
        return "Untitled"


def writer_agent(state):
    """Writer agent per workflow"""
    writer = WriterAgent()
    
    post = writer.write_post(
        topic=state.get('current_topic'),
        research_results=state.get('research_results', {}),
        max_length=state.get('max_post_length', 1500)
    )
    
    return {"draft_post": post}
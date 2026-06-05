# research_agent.py
import ollama
from tools.tavily_search import web_search
import trafilatura
from typing import List, Dict, Any

class ResearchAgent:
    def __init__(self, llm_model: str = "llama3.1"):
        self.llm = ollama.Chat(model=llm_model)
    
    def search_and_extract(self, query: str, max_results: int = 5) -> List[Dict]:
        """Cerca e pulisce i contenuti dalle pagine web"""
        search_results = web_search(query, max_results)
        
        cleaned_docs = []
        for result in search_results.get('results', []):
            url = result.get('url')
            content = trafilatura.extract(trafilatura.fetch_url(url))
            
            if content:
                cleaned_docs.append({
                    'url': url,
                    'title': result.get('title'),
                    'content': content[:5000],  # Limita lunghezza
                    'score': result.get('score')
                })
        
        return cleaned_docs
    
    def react_research(self, topic: str, max_iterations: int = 3) -> Dict:
        """
        ReAct-style research agent
        Thought -> Action -> Observation cycle
        """
        context = {
            'topic': topic,
            'findings': [],
            'queries_used': [],
            'iteration': 0
        }
        
        prompt_template = """
        You are a research agent using ReAct reasoning.
        
        Topic: {topic}
        
        Previous findings: {findings}
        
        Thought: Consider what information you need and what you already know.
        Action: Decide what search query to execute next.
        
        Respond in format:
        THOUGHT: [your reasoning]
        ACTION: [search query]
        """
        
        for i in range(max_iterations):
            # Genera prossima azione
            response = self.llm.invoke(
                prompt_template.format(
                    topic=topic,
                    findings=context['findings']
                )
            )
            
            # Parse response (semplificato)
            if "ACTION:" in response.content:
                query = response.content.split("ACTION:")[1].strip()
                context['queries_used'].append(query)
                
                # Esegui ricerca
                results = self.search_and_extract(query)
                
                # Observation
                obs_prompt = f"""
                Search results for "{query}":
                {results}
                
                Observation: Summarize key findings and what's still missing.
                """
                obs_response = self.llm.invoke(obs_prompt)
                
                context['findings'].append({
                    'iteration': i,
                    'query': query,
                    'observation': obs_response.content,
                    'documents': results
                })
                
                context['iteration'] = i + 1
        
        # Sintesi finale
        synthesis_prompt = f"""
        Based on all research findings for topic "{topic}":
        
        {context['findings']}
        
        Provide a comprehensive research summary with:
        1. Key facts discovered
        2. Gaps in information
        3. Recommended sources to cite
        
        Format as structured research report.
        """
        
        synthesis = self.llm.invoke(synthesis_prompt)
        
        return {
            'topic': topic,
            'research_summary': synthesis.content,
            'findings': context['findings'],
            'sources': [doc for finding in context['findings'] 
                       for doc in finding.get('documents', [])]
        }


# Versione base per integrazione
def research_agent(state):
    """Research agent con ReAct per workflow"""
    topic = state.get('current_topic')
    
    if not topic:
        return {"research_results": "No topic specified"}
    
    researcher = ResearchAgent()
    results = researcher.react_research(topic)
    
    return {"research_results": results}
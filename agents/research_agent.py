# agents/research_agent.py
import ollama
from tools.tavily_search import web_search
import trafilatura
from typing import List, Dict, Any
import json
import re

class ResearchAgent:
    def __init__(self, llm_model: str = "llama3.1"):
        self.llm_model = llm_model
    
    def _call_llm(self, prompt: str) -> str:
        """Chiamata corretta a Ollama"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return ""
    
    def search_and_extract(self, query: str, max_results: int = 3) -> List[Dict]:
        """Cerca e pulisce i contenuti dalle pagine web"""
        print(f"  🔍 Searching: {query}")
        
        try:
            search_results = web_search.invoke({"query": query, "max_results": max_results})
        except Exception as e:
            print(f"  ❌ Search error: {e}")
            return []
        
        cleaned_docs = []
        results = search_results.get('results', [])
        print(f"  📊 Got {len(results)} raw results")
        
        for result in results[:max_results]:
            url = result.get('url')
            title = result.get('title', 'No title')
            print(f"  📄 Extracting: {title[:50]}...")
            
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    content = trafilatura.extract(downloaded)
                    if content and len(content) > 200:
                        cleaned_docs.append({
                            'url': url,
                            'title': title,
                            'content': content[:3000],  # Limita per performance
                            'score': result.get('score', 0)
                        })
                        print(f"    ✅ Extracted {len(content)} chars")
                    else:
                        print(f"    ⚠️ Content too short or empty")
                else:
                    print(f"    ⚠️ Could not fetch URL")
            except Exception as e:
                print(f"    ❌ Extraction error: {e}")
        
        return cleaned_docs
    
    def react_research(self, topic: str, max_iterations: int = 2) -> Dict:
        """
        ReAct-style research agent semplificato
        """
        print(f"\n🧠 Researching topic: '{topic}'")
        
        if not topic or topic == "None":
            return {
                'topic': topic,
                'research_summary': f"No valid topic provided. Please specify a topic to research.",
                'findings': [],
                'sources': [],
                'num_sources': 0,
                'queries_used': [],
                'error': True
            }
        
        # Genera query di ricerca specifiche
        query_prompt = f"""
        Generate 2 specific Google search queries to find information about: "{topic}"
        
        Return ONLY a JSON array of strings.
        Example: ["query 1", "query 2"]
        
        Make queries specific and likely to find practical information.
        """
        
        response = self._call_llm(query_prompt)
        
        # Estrai query
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                queries = json.loads(match.group())
            else:
                queries = [topic, f"{topic} guide how to"]
        except:
            queries = [topic, f"{topic} tutorial"]
        
        print(f"📝 Search queries: {queries}")
        
        # Esegui ricerche
        all_documents = []
        findings_list = []
        
        for query in queries[:max_iterations]:
            documents = self.search_and_extract(query, max_results=3)
            
            if documents:
                all_documents.extend(documents)
                
                # Observation
                obs_prompt = f"""
                For topic "{topic}", I searched for "{query}" and found {len(documents)} documents.
                
                Based on these results, what are the key points I should know about {topic}?
                Respond with 2-3 bullet points.
                """
                
                observation = self._call_llm(obs_prompt)
                
                findings_list.append({
                    'query': query,
                    'observation': observation,
                    'num_docs': len(documents)
                })
        
        # Rimuovi duplicati
        unique_docs = {}
        for doc in all_documents:
            if doc['url'] not in unique_docs:
                unique_docs[doc['url']] = doc
        all_documents = list(unique_docs.values())
        
        # Sintesi finale
        if all_documents:
            # Prepara riassunto fonti
            sources_text = "\n".join([
                f"Source: {doc['title']}\nURL: {doc['url']}\nExcerpt: {doc['content'][:500]}...\n---"
                for doc in all_documents[:3]
            ])
            
            synthesis_prompt = f"""
            Topic: {topic}
            
            Research findings from {len(all_documents)} sources:
            
            {sources_text}
            
            Provide a comprehensive research summary (300-500 words) for writing a blog post.
            Include:
            1. Key facts and main points
            2. Practical tips or advice (if any)
            3. Important context or background
            
            Write in clear paragraphs.
            """
            
            summary = self._call_llm(synthesis_prompt)
        else:
            summary = f"No reliable sources found for '{topic}'. This could be because:\n1. The topic is too specific\n2. Tavily API key is invalid\n3. No internet connection\n\nSuggestions: Try a broader topic or check your API configuration."
        
        return {
            'topic': topic,
            'research_summary': summary,
            'findings': findings_list,
            'sources': all_documents,
            'num_sources': len(all_documents),
            'queries_used': queries,
            'error': False
        }


def research_agent(state):
    """Research agent con ReAct per workflow"""
    print("\n" + "="*50)
    print("🔍 RESEARCH AGENT STARTING")
    print("="*50)
    
    # Prende il topic dallo state
    topic = state.get('current_topic')
    
    print(f"📚 Current topic from state: '{topic}'")
    
    if not topic or topic == "None":
        print("❌ ERROR: No valid topic found in state!")
        print("Available state keys:", state.keys())
        return {
            "research_results": {
                "error": True,
                "topic": None,
                "research_summary": "No topic specified. The planner agent did not set a current_topic.",
                "sources": [],
                "num_sources": 0
            }
        }
    
    researcher = ResearchAgent()
    results = researcher.react_research(topic)
    
    print(f"\n✅ Research complete: {results['num_sources']} sources found")
    if results['num_sources'] == 0:
        print("⚠️ WARNING: No sources found! Check Tavily API key.")
    print("="*50)
    
    return {"research_results": results}
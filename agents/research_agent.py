# agents/research_agent.py (modificato con K-RAG)
import ollama
from tools.tavily_search import web_search
from tools.rag_retriever import rag_retrieve, rag_add_documents
from tools.kg_tool import get_kg_tool
import trafilatura
from typing import List, Dict, Any
import json
import re

class ResearchAgent:
    def __init__(self, llm_model: str = "llama3.1", kg_manager=None):
        self.llm_model = llm_model
        self.kg_manager = kg_manager
        self.kg_tool = get_kg_tool(kg_manager) if kg_manager else None
    
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
    
    def _expand_query_with_kg(self, topic: str) -> str:
        """Espande la query usando il Knowledge Graph"""
        if not self.kg_tool:
            return topic
        
        try:
            # Trova topic correlati dal KG
            related = self.kg_tool.get_related(topic, depth=1)
            if related:
                expanded = f"{topic} related topics: {', '.join(related[:3])}"
                print(f"🔍 KG Query Expansion: '{topic}' -> related: {related[:3]}")
                return expanded
        except Exception as e:
            print(f"⚠️ KG expansion error: {e}")
        
        return topic
    
    def search_and_extract(self, query: str, max_results: int = 1) -> List[Dict]:
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
                            'content': content[:3000],
                            'score': result.get('score', 0)
                        })
                        print(f"    ✅ Extracted {len(content)} chars")
            except Exception as e:
                print(f"    ❌ Extraction error: {e}")
        
        return cleaned_docs
    
    def react_research(self, topic: str, max_iterations: int = 2) -> Dict:
        """
        ReAct-style research agent con K-RAG
        Thought -> Action -> Observation cycle
        """
        print(f"\n🧠 ReAct Research with K-RAG for: '{topic}'")
        
        # STEP 1: Expand query with KG
        kg_context = None
        if self.kg_tool:
            kg_context = self.kg_tool.get_related(topic)
            print(f"📊 KG Context: Related topics = {kg_context[:3] if kg_context else 'None'}")
        
        # STEP 2: Query RAG vector store first
        print("\n📚 STEP 1: Querying RAG vector store...")
        rag_docs = []
        try:
            rag_docs = rag_retrieve(topic, k=3, kg_context=", ".join(kg_context[:3]) if kg_context else None)
            if rag_docs:
                print(f"✅ Found {len(rag_docs)} relevant documents in RAG store")
        except Exception as e:
            print(f"⚠️ RAG retrieval error: {e}")
        
        # STEP 3: Generate search queries (informed by KG and RAG)
        kg_hint = f"Related topics from KG: {', '.join(kg_context[:3])}" if kg_context else ""
        rag_hint = f"Already have information on: {rag_docs[0]['metadata'].get('title', 'N/A')[:100]}" if rag_docs else ""
        
        query_prompt = f"""
        Generate 2 specific search queries to find NEW information about: "{topic}"
        
        {kg_hint}
        {rag_hint}
        
        Return ONLY a JSON array of strings.
        Example: ["query 1", "query 2"]
        """
        
        response = self._call_llm(query_prompt)
        
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                queries = json.loads(match.group())
            else:
                queries = [topic, f"{topic} guide"]
        except:
            queries = [topic, f"{topic} tutorial"]
        
        print(f"📝 Search queries: {queries}")
        
        # STEP 4: Execute web searches
        all_documents = []
        
        # Aggiungi documenti dal RAG come "virtual sources"
        for doc in rag_docs:
            all_documents.append({
                'url': doc['metadata'].get('url', 'rag://local'),
                'title': doc['metadata'].get('title', 'RAG Document'),
                'content': doc['content'],
                'score': doc['relevance_score'],
                'source': 'rag'
            })
        
        # Web search
        for query in queries[:max_iterations]:
            documents = self.search_and_extract(query, max_results=3)
            if documents:
                all_documents.extend(documents)
        
        # STEP 5: Add new documents to RAG for future use
        if all_documents:
            try:
                rag_add_documents(all_documents)
                print(f"💾 Added {len(all_documents)} documents to RAG store")
            except Exception as e:
                print(f"⚠️ Could not add to RAG: {e}")
        
        # Rimuovi duplicati
        unique_docs = {}
        for doc in all_documents:
            if doc['url'] not in unique_docs:
                unique_docs[doc['url']] = doc
        all_documents = list(unique_docs.values())
        
        # STEP 6: Synthesis with KG context
        if all_documents:
            sources_text = "\n".join([
                f"Source: {doc['title']}\nURL: {doc['url']}\nExcerpt: {doc['content'][:500]}..."
                for doc in all_documents[:3]
            ])
            
            synthesis_prompt = f"""
            Topic: {topic}
            
            Knowledge Graph Context (related topics): {', '.join(kg_context[:3]) if kg_context else 'None'}
            
            Research findings from {len(all_documents)} sources:
            
            {sources_text}
            
            Provide a comprehensive research summary (300-500 words) for writing a blog post.
            Include:
            1. Key facts and main points
            2. Practical tips or advice
            3. How this relates to broader context
            
            Write in clear paragraphs.
            """
            
            summary = self._call_llm(synthesis_prompt)
        else:
            summary = f"No reliable sources found for '{topic}'."
        
        return {
            'topic': topic,
            'research_summary': summary,
            'sources': all_documents,
            'num_sources': len(all_documents),
            'rag_sources_count': len(rag_docs),
            'queries_used': queries,
            'kg_context': kg_context,
            'error': False
        }


def research_agent(state):
    """Research agent con K-RAG per workflow"""
    print("\n" + "="*50)
    print("🔍 RESEARCH AGENT with K-RAG STARTING")
    print("="*50)
    
    topic = state.get('current_topic')
    kg_manager = state.get('kg_manager')
    
    print(f"📚 Current topic: '{topic}'")
    
    if not topic or topic == "None":
        return {
            "research_results": {
                "error": True,
                "topic": None,
                "research_summary": "No topic specified.",
                "sources": [],
                "num_sources": 0
            }
        }
    
    researcher = ResearchAgent(kg_manager=kg_manager)
    results = researcher.react_research(topic)
    
    print(f"\n✅ Research complete:")
    print(f"   - Web sources: {results['num_sources']}")
    print(f"   - RAG sources: {results['rag_sources_count']}")
    print("="*50)
    
    return {"research_results": results}
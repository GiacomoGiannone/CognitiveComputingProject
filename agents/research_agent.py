# agents/research_agent.py (modificato con K-RAG)
import ollama
from tools.tavily_search import web_search
from tools.rag_retriever import rag_retrieve, rag_add_documents
from tools.kg_tool import get_kg_tool
from tools.entity_extractor import extract_topics_from_title
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
    
    def _expand_query_with_kg(self, topic: str, extracted_topics: list = None, blog_domain: str = None) -> str:
        """
        Espande la query usando il Knowledge Graph.
        
        MIGLIORAMENTO ARCHITETTURALE:
        Usa i topic generici già estratti (una sola volta dal planner),
        anziché re-estrarre dal topic specifico ogni volta.
        
        Args:
            topic: Il topic specifico
            extracted_topics: Topic generici già estratti dal planner (PASSATI DALLO STATE)
            blog_domain: Il dominio del blog (opzionale, fallback se no extracted_topics)
        
        Returns:
            Query espansa con topic correlati dal KG
        """
        if not self.kg_tool:
            return topic
        
        try:
            # 🔧 Se i topic sono già nel state, usali direttamente
            if extracted_topics:
                print(f"\n🔍 KG Query Expansion (usando topic dallo state)")
                print(f"   Original specific topic: {topic[:60]}...")
                print(f"   Using pre-extracted topics: {extracted_topics}")
            else:
                # Fallback: estrai se non disponibili
                print(f"\n🔍 KG Query Expansion (estrazione fallback)")
                print(f"   Original specific topic: {topic[:60]}...")
                extracted_topics = extract_topics_from_title(topic, blog_domain)
                print(f"   Extracted generic topics: {extracted_topics}")
            
            # STEP 2: Usa i topic estratti per cercare correlati nel KG
            all_related = []
            for generic_topic in extracted_topics:
                try:
                    related = self.kg_tool.get_related(generic_topic, depth=1)
                    if related:
                        all_related.extend(related)
                        print(f"   - {generic_topic} -> {related[:2]}")
                except Exception as e:
                    print(f"   ⚠️ Could not find relations for '{generic_topic}': {e}")
            
            # STEP 3: Rimuovi duplicati e crea query espansa
            unique_related = list(set(all_related))[:3]
            if unique_related:
                expanded = f"{topic}\nRelated topics to consider: {', '.join(unique_related)}"
                print(f"   ✅ KG Expansion complete: +{len(unique_related)} related topics")
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
    
    def react_research(self, topic: str, extracted_topics: list = None, blog_domain: str = None, max_iterations: int = 2) -> Dict:
        """
        ReAct-style research agent con K-RAG
        Thought -> Action -> Observation cycle
        
        🔧 MIGLIORAMENTO: Usa topic generici già estratti dal planner (state)
        """
        print(f"\n🧠 ReAct Research with K-RAG for: '{topic}'")
        
        # STEP 1: Expand query with KG using pre-extracted topics
        kg_context = None
        if self.kg_tool:
            # 🔧 Usa i topic generici già estratti (passati dal state)
            expanded_query = self._expand_query_with_kg(topic, extracted_topics, blog_domain)
            
            # Usa i topic estratti per il contesto KG
            topics_for_kg = extracted_topics if extracted_topics else []
            kg_context = []
            for gt in topics_for_kg:
                try:
                    related = self.kg_tool.get_related(gt, depth=1)
                    if related:
                        kg_context.extend(related)
                except:
                    pass
            
            if kg_context:
                kg_context = list(set(kg_context))[:5]  # Rimuovi duplicati
                print(f"📊 KG Context: Related topics = {kg_context[:3]}")
            else:
                print(f"📊 KG Context: No related topics found")
        
        # STEP 2: Query RAG vector store first
        print("\n📚 Querying RAG vector store...")
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
    blog_domain = state.get('blog_domain')  
    extracted_topics = state.get('extracted_graph_topics')  
    
    print(f"📚 Current topic: '{topic}'")
    print(f"📌 Blog domain: '{blog_domain}'")
    print(f"🔧 Pre-extracted topics: {extracted_topics}") 
    
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
    results = researcher.react_research(topic, extracted_topics=extracted_topics, blog_domain=blog_domain)  
    
    print(f"\n✅ Research complete:")
    print(f"   - Web sources: {results['num_sources']}")
    print(f"   - RAG sources: {results['rag_sources_count']}")
    print("="*50)
    
    return {"research_results": results}
# agents/research_agent.py
"""
Research Agent con approccio ReAct (Thought → Action → Observation).
L'LLM (qwen3) decide autonomamente quali tool usare, con quali parametri
e quante volte, tramite bind_tools di LangChain.
"""
from langsmith import traceable
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from tools.tavily_search import web_search
from tools.rag_tool import rag_search
from tools.kg_tool import kg_search, get_kg_tool
from tools.rag_retriever import rag_add_documents, rag_retrieve
from typing import Dict, Any


# ─── System Prompt per il Research Agent ───────────────────────────────────────

RESEARCH_SYSTEM_PROMPT = """You are an expert Research Agent following the ReAct paradigm (Thought → Action → Observation).
Your goal is to gather comprehensive, accurate, and diverse information about a given topic to support writing a high-quality blog post.

CRITICAL RULE — ALWAYS THINK BEFORE ACTING:
Before EVERY tool call, you MUST write a brief "Thought:" explaining:
- What information you still need
- Why you chose this specific tool and query
- How this fits your overall research strategy
Never call a tool without explaining your reasoning first.

You have access to the following tools:

1. **rag_search**: Search the internal vector store for previously collected documents. 
   Use this FIRST to check if relevant information already exists.
   
2. **kg_search**: Search the Knowledge Graph for related topics and connections.
   Use this to discover broader context, related subjects, and previously covered topics.
   
3. **web_search**: Search the web for recent and up-to-date information.
   Use this to find NEW information not already in the RAG store, or to get current data.

STRATEGY:
1. Start by checking the Knowledge Graph (kg_search) for ALL of the pre-extracted topics and keywords at once (by passing them as a comma-separated list, e.g., "Golf, Technique, Improvement") to discover related topics and connections.
2. Use the related topics found in the KG to EXPAND your queries: when you search the RAG store 
   or the web, incorporate the related topics from the KG into your search queries to get 
   broader and more relevant results.
3. Search the RAG store (rag_search) using both the original topic AND the KG-expanded terms.
4. Search the web (web_search) to fill gaps, using KG-related topics to diversify your queries.
5. You may call tools multiple times with different queries if needed.
6. When you have gathered enough information (key facts, practical tips, broader context), 
   stop calling tools and provide your final research summary.

EXAMPLE of KG-expanded research:
- Topic: "Preparazione fisica per il nuoto"
- KG returns related topics: "Resistenza", "Allenamento Funzionale", "Sport Acquatici"
- Then search RAG/web with expanded queries like:
  "preparazione fisica nuoto resistenza allenamento funzionale"
  "sport acquatici endurance training"

FINAL OUTPUT:
When you are satisfied with the information gathered, respond with a comprehensive 
research summary (300-500 words) that includes:
1. Key facts and main points about the topic
2. Practical tips or advice
3. How this topic relates to broader context
4. Sources and references found

Write the summary in clear paragraphs. Do NOT call any more tools when producing the final summary."""


# ─── Helpers per formattare le Observation nel log ─────────────────────────────

@traceable(name="FormatObservation", run_type="chain", tags=["helper", "research"])
def _format_observation(tool_name: str, tool_result) -> str:
    """Formatta l'observation per il log in modo leggibile e conciso."""

    #ad esempio:
    # RAG results:
    #   [1] Title of Document 1 (Score: 0.95):
    #       sdfgjdfngjdfngjdfngjgfnd 
    #       .....................
    # diventa solo il titolo e lo score
    if tool_name == "rag_search":
        result_str = str(tool_result)
        # Estrai solo i titoli dai risultati RAG
        lines = result_str.split('\n')
        titles = [l.strip() for l in lines if l.strip().startswith('[')]
        if titles:
            # Mostra solo la riga con titolo e score (prima riga di ogni blocco)
            title_lines = []
            for t in titles:
                # Prendi solo fino alla fine della riga del titolo (prima di \n con contenuto)
                title_lines.append(t.split('\n')[0])
            return "RAG results:\n   " + "\n   ".join(title_lines)
        return result_str[:200]

    # web search invece costruisce:
    #   titolo
    #   url
    elif tool_name == "web_search":
        if isinstance(tool_result, dict):
            results = tool_result.get('results', [])
            if results:
                formatted = []
                for i, r in enumerate(results, 1):
                    title = r.get('title', 'N/A')[:80]
                    url = r.get('url', '')
                    score = r.get('score', 0.0)
                    formatted.append(f"[{i}] {title} (Score: {score:.3f})\n       {url}")
                return f"Web results ({len(results)} found):\n   " + "\n   ".join(formatted)
        return str(tool_result)[:300]

    elif tool_name == "kg_search":
        return str(tool_result)

    else:
        return str(tool_result)[:300]


# ─── Funzione principale per il workflow LangGraph ────────────────────────────

@traceable(name="ResearchAgent", run_type="chain", tags=["agent", "research", "react"])
def research_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research agent con loop ReAct custom per workflow LangGraph.
    
    Usa ChatOllama (qwen3) con bind_tools per decidere autonomamente
    quali tool chiamare durante la ricerca.
    """
    print("\n" + "=" * 50)
    print("🔍 RESEARCH AGENT (ReAct) STARTING")
    print("=" * 50)

    topic = state.get('current_topic')
    kg_manager = state.get('kg_manager')
    blog_domain = state.get('blog_domain')
    extracted_topics = state.get('extracted_graph_topics', [])

    print(f"📚 Current topic: '{topic}'")
    print(f"📌 Blog domain: '{blog_domain}'")
    print(f"🔧 Pre-extracted topics: {extracted_topics}")

    # ── Caso: nessun topic ──
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

    # ── Inizializza KG tool singleton se disponibile ──
    if kg_manager:
        get_kg_tool(kg_manager)
        print("✅ Knowledge Graph tool initialized")

    # ── Setup LLM con tools (kg_search solo se KG disponibile) ──
    tools = [web_search, rag_search]
    if kg_manager:
        tools.append(kg_search)
    tool_map = {t.name: t for t in tools}

    llm = ChatOllama(model="qwen3", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"🤖 LLM: qwen3 with {len(tools)} tools bound: {list(tool_map.keys())}")

    # ── Costruisci contesto iniziale ──
    topic_context = f"Research this topic thoroughly: \"{topic}\""
    if extracted_topics:
        topic_context += f"\n\nRelated generic topics (from Knowledge Graph extraction): {', '.join(extracted_topics)}"
    if blog_domain:
        topic_context += f"\nBlog domain: {blog_domain}"

    # ── Messaggi iniziali ──
    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=topic_context),
    ]

    # ── Loop ReAct custom ──
    max_iterations = 5
    research_summary = ""
    tool_calls_log = []    # Log di tutte le tool call effettuate
    collected_sources = [] # Sorgenti web raccolte per rag_add_documents
    rag_docs_found = 0     # Contatore documenti RAG trovati
    kg_topics_found = 0    # Contatore topic KG trovati

    print(f"\n🔄 Starting ReAct loop (max {max_iterations} iterations)...\n")

    for iteration in range(max_iterations):
        print(f"\n{'─' * 50}")
        print(f"🔄 Iteration {iteration + 1}/{max_iterations}")
        print(f"{'─' * 50}")

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f"❌ LLM invocation error: {e}")
            research_summary = f"LLM error during research: {str(e)}"
            break

        messages.append(response)

        raw_content = response.content or ""

        # Tenta di estrarre il pensiero dai tag nativi <think> prima di pulire la stringa
        think_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
        if think_match:
            thought_text = think_match.group(1).strip()
        else:
            thought_text = raw_content.strip()

        # Pulisci il contenuto rimuovendo i tag per i controlli successivi (es. Final Answer)
        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

        # Controlla se l'LLM ha deciso di terminare la ricerca
        # Finisce quando pensa di avere abbastanza informazioni
        if not response.tool_calls:
            research_summary = clean_content
            print(f"\n📝 Final Answer:\n{research_summary}")
            print(f"\n✅ ReAct loop completed early at iteration {iteration + 1}/{max_iterations} — LLM has enough information")
            break

        # Genera una giustificazione dinamica se Ollama ha azzerato il testo (Native Tool Calling constraint)
        # Praticamente e' solo un filler per mostrare un thought logico e contestuale, anche se l'LLM non lo ha generato
        if not thought_text and response.tool_calls:
            primary_tool = response.tool_calls[0]["name"]
            tool_args = response.tool_calls[0]["args"]
            # Estraiamo la query per rendere il thought altamente specifico e contestuale
            query_param = tool_args.get("query") or tool_args.get("topic") or ""
            thought_text = f"Analizzo lo stato della ricerca. Per approfondire il topic, decido di invocare lo strumento '{primary_tool}' con focus su: '{query_param}'."

        # THOUGHT: Mostra a schermo il vero pensiero logico o la giustificazione d'uso del tool
        print(f"\n💭 Thought: {thought_text}")

        # ── ACTION + OBSERVATION per ogni tool call ──
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            print(f"\n⚡ Action: {tool_name}({tool_args})")
            tool_calls_log.append({"tool": tool_name, "args": tool_args})

            try:
                if tool_name in tool_map:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                    result_str = str(tool_result)

                    # Raccogli sorgenti per il writer e per rag_add_documents
                    # Serviranno al writer agent per citare le fonti ed evitare duplicati
                    if tool_name == "web_search" and isinstance(tool_result, dict):
                        for r in tool_result.get('results', []):
                            url = r.get('url', '')
                            # Evita duplicati in collected_sources
                            if url and not any(src['url'] == url for src in collected_sources):
                                collected_sources.append({
                                    'url': url,
                                    'title': r.get('title', ''),
                                    'content': r.get('content', '')[:3000],
                                    'source': 'web_search',
                                    'relevance_score': r.get('score', 0.0)
                                })
                    elif tool_name == "rag_search":
                        # Recupera dati strutturati RAG per passarli al writer
                        query_used = tool_args.get('query', '')
                        rag_structured = rag_retrieve(query_used, k=5)
                        for doc in rag_structured:
                            url = doc['metadata'].get('url', 'rag://local')
                            # Evita duplicati in collected_sources
                            if url and not any(src['url'] == url for src in collected_sources):
                                collected_sources.append({
                                    'url': url,
                                    'title': doc['metadata'].get('title', 'RAG Document'),
                                    'content': doc['content'],
                                    'source': 'rag',
                                    'relevance_score': doc.get('relevance_score', 0.0)
                                })

                    # Traccia contatori per RAG e KG
                    if tool_name == "rag_search":
                        # Conta i risultati trovati (ogni blocco inizia con [N])
                        rag_docs_found += len(re.findall(r'^\[\d+\]', result_str, re.MULTILINE))
                    elif tool_name == "kg_search":
                        # Conta topic correlati trovati
                        if "Topics related to" in result_str:
                            topics_part = result_str.split(": ", 1)[-1]
                            kg_topics_found += len(topics_part.split(", "))

                    # Formatta observation in modo leggibile
                    formatted_obs = _format_observation(tool_name, tool_result)
                    print(f"👁️ Observation: {formatted_obs}")
                else:
                    result_str = f"Error: tool '{tool_name}' not found."
                    print(f"👁️ Observation: ❌ {result_str}")
            except Exception as e:
                result_str = f"Tool execution error: {str(e)}"
                print(f"👁️ Observation: ❌ {result_str}")

            # Aggiungi il risultato come ToolMessage nella cronologia
            messages.append(ToolMessage(
                content=result_str,
                tool_call_id=tool_call_id,
            ))

    else:
        # Max iterazioni raggiunte senza summary finale
        print(f"\n⚠️ Max iterations ({max_iterations}) reached without final answer")
        # Prova a estrarre contenuto dall'ultima risposta
        if messages and hasattr(messages[-1], 'content') and messages[-1].content:
            raw = messages[-1].content
            research_summary = re.sub(
                r'<think>.*?</think>', '', raw, flags=re.DOTALL
            ).strip()
        
        if not research_summary:
            # Fallback: chiedi all'LLM di sintetizzare senza tool
            print("📝 Requesting final summary from LLM...")
            try:
                fallback_llm = ChatOllama(model="qwen3", temperature=0)
                messages.append(HumanMessage(
                    content="Please provide your final research summary now, "
                            "based on all the information gathered so far. "
                            "Do NOT call any tools. Do NOT include <think> tags."
                ))
                fallback_response = fallback_llm.invoke(messages)
                raw = fallback_response.content or ""
                research_summary = re.sub(
                    r'<think>.*?</think>', '', raw, flags=re.DOTALL
                ).strip()
            except Exception as e:
                research_summary = f"Could not generate summary: {str(e)}"

    # Ordina tutte le sorgenti raccolte per pertinenza decrescente
    collected_sources.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)

    # ── Post-loop: aggiungi solo i nuovi documenti web al RAG store ──
    web_sources = [src for src in collected_sources if src.get('source') == 'web_search']
    if web_sources:
        try:
            added = rag_add_documents(web_sources)
            print(f"💾 Post-loop: Added {added} chunks from {len(web_sources)} unique web sources to RAG store")
        except Exception as e:
            print(f"⚠️ Could not add documents to RAG: {e}")

    # ── Risultato finale ──
    results = {
        'topic': topic,
        'research_summary': research_summary,
        'sources': collected_sources,
        'num_sources': len(collected_sources),
        'rag_docs_found': rag_docs_found,
        'kg_topics_found': kg_topics_found,
        'tool_calls_log': tool_calls_log,
        'iterations_used': iteration + 1 if 'iteration' in dir() else 0,
        'error': False
    }

    print(f"\n✅ Research complete:")
    print(f"   - Total tool calls: {len(tool_calls_log)}")
    print(f"   - RAG documents retrieved: {rag_docs_found}")
    print(f"   - KG related topics found: {kg_topics_found}")
    print(f"   - Web sources collected: {len(web_sources)}")
    print(f"   - Total unique sources (for writer): {results['num_sources']}")
    print(f"   - Iterations used: {results['iterations_used']}/{max_iterations}")
    print(f"   - Summary length: {len(research_summary)} chars")
    print("=" * 50)

    return {"research_results": results}
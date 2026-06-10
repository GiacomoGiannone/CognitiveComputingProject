# tools/rag_tool.py
"""
RAG Tool - Interfaccia @tool LangChain per il RAG vector store.
Wrappa rag_retriever.py per esporre una funzione compatibile con bind_tools.
"""
from langchain.tools import tool
from tools.rag_retriever import rag_retrieve


@tool
def rag_search(query: str) -> str:
    """Search the internal RAG vector store for previously collected documents
    relevant to the query. Use this tool to check if there is already useful
    information on a topic before searching the web. Returns the most relevant
    document excerpts with titles and relevance scores."""
    try:
        docs = rag_retrieve(query, k=5)
        if not docs:
            return "No relevant documents found in the RAG store."

        results = []
        for i, doc in enumerate(docs, 1):
            title = doc['metadata'].get('title', 'Unknown')
            content = doc['content'][:500]
            score = doc.get('relevance_score', 0)
            results.append(
                f"[{i}] Title: {title} (relevance: {score:.2f})\n{content}"
            )

        return "\n\n".join(results)
    except Exception as e:
        return f"RAG retrieval error: {str(e)}"
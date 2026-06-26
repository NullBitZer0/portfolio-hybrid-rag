import hashlib
import time
from langchain_core.tools import tool
from src.opensearch_client import get_opensearch_client, hybrid_search, get_doc_count
from src.embeddings import embed_query
from src.retrieval import cohere_rerank, RERANK_TOP_K
from src.query_expansion import expand_query
from langchain_core.documents import Document


# ── Tool Result Cache (in-memory, 5min TTL) ────────────────

_tool_cache: dict[str, dict] = {}
TOOL_CACHE_TTL = 300  # 5 minutes


def _cache_key(query: str, source_filter: str = None) -> str:
    content = f"{query}|{source_filter or ''}"
    return hashlib.md5(content.encode()).hexdigest()


def _get_cached(key: str) -> str | None:
    entry = _tool_cache.get(key)
    if entry and time.time() - entry["ts"] < TOOL_CACHE_TTL:
        return entry["result"]
    if entry:
        del _tool_cache[key]
    return None


def _set_cached(key: str, result: str):
    if len(_tool_cache) > 200:
        oldest = min(_tool_cache, key=lambda k: _tool_cache[k]["ts"])
        del _tool_cache[oldest]
    _tool_cache[key] = {"result": result, "ts": time.time()}


# ── Search Helper ──────────────────────────────────────────

def _search(query: str, source_filter: str = None, k: int = 10) -> str:
    """Internal search helper with query expansion, Cohere reranking, and caching."""
    key = _cache_key(query, source_filter)
    cached = _get_cached(key)
    if cached is not None:
        print(f"Tool cache hit: {query[:40]}...")
        return cached

    client = get_opensearch_client()
    expanded = expand_query(query)
    query_vector = embed_query(expanded)
    results = hybrid_search(client, expanded, query_vector, k=k, source_filter=source_filter)

    if not results:
        return "No relevant documents found."

    docs = [
        Document(
            page_content=r["content"],
            metadata={"source": r["source"], "page": r["page"], "score": r["score"]},
        )
        for r in results
    ]

    docs = cohere_rerank(query, docs, top_n=RERANK_TOP_K)

    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    result = "\n\n---\n\n".join(formatted)
    _set_cached(key, result)
    return result


@tool
def search_all(query: str) -> str:
    """Search all documents for any information about Adeesha Perera's portfolio, projects, skills, experience, or education."""
    return _search(query)


@tool
def search_projects(query: str) -> str:
    """Search specifically for project information. Use this when the user asks about projects Adeesha has built or worked on."""
    client = get_opensearch_client()
    expanded = expand_query(query)
    query_vector = embed_query(expanded)

    all_results = []
    for source_name in ["realtime_fraud_detection.pdf", "hybrid_rag_project.pdf", "resume/all_projects.pdf"]:
        results = hybrid_search(client, expanded, query_vector, k=5, source_filter=source_name)
        all_results.extend(results)

    if not all_results:
        return "No project documents found."

    all_results.sort(key=lambda x: x["score"], reverse=True)
    all_results = all_results[:8]

    docs = [
        Document(
            page_content=r["content"],
            metadata={"source": r["source"], "page": r["page"], "score": r["score"]},
        )
        for r in all_results
    ]

    docs = cohere_rerank(query, docs, top_n=RERANK_TOP_K)

    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


@tool
def search_skills(query: str) -> str:
    """Search for ALL skills information about Adeesha. Use this when the user asks about technical skills, tools, technologies, programming languages, ML frameworks, soft skills, or ANY skills-related question. Searches both technical_skills.pdf (Python, PyTorch, TensorFlow, Docker, AWS, etc.) and soft_skills.pdf (leadership, teamwork, etc.)."""
    client = get_opensearch_client()
    expanded = expand_query(query)
    query_vector = embed_query(expanded)

    all_results = []
    for source_name in ["resume/technical_skills.pdf", "soft_skills.pdf"]:
        results = hybrid_search(client, expanded, query_vector, k=5, source_filter=source_name)
        all_results.extend(results)

    if not all_results:
        return "No skills documents found."

    all_results.sort(key=lambda x: x["score"], reverse=True)
    all_results = all_results[:8]

    docs = [
        Document(
            page_content=r["content"],
            metadata={"source": r["source"], "page": r["page"], "score": r["score"]},
        )
        for r in all_results
    ]

    docs = cohere_rerank(query, docs, top_n=RERANK_TOP_K)

    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


@tool
def search_source(query: str, source: str) -> str:
    """Search within a specific document file. Use this when you need details from a particular document. The source must be an exact filename like 'realtime_fraud_detection.pdf'."""
    return _search(query, source_filter=source)


@tool
def list_documents() -> str:
    """List all available documents in the knowledge base with their chunk counts. Use this to understand what information is available before searching."""
    client = get_opensearch_client()
    try:
        response = client.search(
            index="rag-document",
            body={
                "size": 0,
                "aggs": {
                    "sources": {
                        "terms": {"field": "source", "size": 50}
                    }
                },
            },
        )
        buckets = response["aggregations"]["sources"]["buckets"]
        if not buckets:
            return "No documents indexed."

        lines = ["Available documents:"]
        for b in buckets:
            lines.append(f"- {b['key']}: {b['doc_count']} chunks")
        return "\n".join(lines)
    except Exception:
        return "Could not list documents."

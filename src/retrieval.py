from dataclasses import dataclass
from langchain_core.documents import Document
from src.config import RETRIEVAL_K, COHERE_API_KEY, USE_HYDE


RERANK_TOP_K = 3


@dataclass
class OpenSearchRetriever:
    """Retriever that wraps OpenSearch hybrid search and returns LangChain Documents."""
    k: int = RETRIEVAL_K
    source_filter: str = None

    def invoke(self, query: str) -> list[Document]:
        from src.opensearch_client import get_opensearch_client, hybrid_search
        from src.embeddings import embed_query

        client = get_opensearch_client()

        if USE_HYDE:
            from src.hyde import generate_hypothetical_answer
            search_query = generate_hypothetical_answer(query)
            print(f"HyDE: '{query}' -> '{search_query[:80]}...'")
        else:
            search_query = query

        query_vector = embed_query(search_query)
        results = hybrid_search(client, search_query, query_vector, k=self.k, source_filter=self.source_filter)

        return [
            Document(
                page_content=r["content"],
                metadata={"source": r["source"], "page": r["page"], "score": r["score"]},
            )
            for r in results
        ]


def cohere_rerank(query: str, docs: list[Document], top_n: int = RERANK_TOP_K) -> list[Document]:
    """Rerank documents using Cohere rerank-v3 API."""
    import cohere

    co = cohere.Client(COHERE_API_KEY)

    results = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=[d.page_content for d in docs],
        top_n=top_n,
        return_documents=True,
    )

    reranked = []
    for result in results.results:
        doc = docs[result.index]
        doc.metadata["rerank_score"] = result.relevance_score
        reranked.append(doc)

    print(f"Cohere reranked {len(docs)} docs -> top {top_n}")
    return reranked


class HybridRetriever:
    """Hybrid retriever: OpenSearch BM25 + k-NN, reranked with Cohere API."""

    def __init__(self, rerank: bool = True, k: int = 10, source_filter: str = None):
        self.base_retriever = OpenSearchRetriever(k=k, source_filter=source_filter)
        self.rerank = rerank
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        if self.rerank and docs:
            docs = cohere_rerank(query, docs, top_n=RERANK_TOP_K)
        return docs


def get_reranked_retriever(rerank: bool = True, source_filter: str = None) -> HybridRetriever:
    """Build the hybrid retriever with Cohere reranking."""
    retriever = HybridRetriever(rerank=rerank, k=10, source_filter=source_filter)
    print("Hybrid retriever built: OpenSearch BM25 + k-NN + Cohere rerank")
    return retriever

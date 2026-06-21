from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.config import RETRIEVAL_K, RERANKER_MODEL, RERANK_TOP_K


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 3


@dataclass
class OpenSearchRetriever:
    """Retriever that wraps OpenSearch hybrid search and returns LangChain Documents."""
    k: int = RETRIEVAL_K

    def invoke(self, query: str) -> list[Document]:
        from src.opensearch_client import get_opensearch_client, hybrid_search
        from src.embeddings import embed_query

        client = get_opensearch_client()
        query_vector = embed_query(query)

        results = hybrid_search(client, query, query_vector, k=self.k)

        return [
            Document(
                page_content=r["content"],
                metadata={"source": r["source"], "page": r["page"], "score": r["score"]},
            )
            for r in results
        ]


def build_reranker(top_k: int = RERANK_TOP_K) -> CrossEncoderReranker:
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_k)
    print(f"Reranker built: {RERANKER_MODEL} (top {top_k})")
    return reranker


class HybridRetriever:
    """Hybrid retriever: OpenSearch BM25 + k-NN, reranked with local cross-encoder."""

    def __init__(self, rerank: bool = True, k: int = 10):
        self.base_retriever = OpenSearchRetriever(k=k)
        self.reranker = build_reranker() if rerank else None
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        if self.reranker and docs:
            docs = self.reranker.compress_documents(docs, query)
        return docs


def get_reranked_retriever(rerank: bool = True) -> HybridRetriever:
    """Build the hybrid retriever with local cross-encoder reranking."""
    retriever = HybridRetriever(rerank=rerank, k=10)
    print("Hybrid retriever built: OpenSearch BM25 + k-NN + cross-encoder rerank")
    return retriever

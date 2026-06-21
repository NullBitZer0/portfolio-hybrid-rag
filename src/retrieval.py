from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.config import RETRIEVAL_K, RERANKER_MODEL, RERANK_TOP_K


@dataclass
class OpenSearchRetriever:
    """Retriever that wraps OpenSearch hybrid search and returns LangChain Documents."""
    k: int = RETRIEVAL_K

    def invoke(self, query: str) -> list[Document]:
        from src.opensearch_client import get_opensearch_client, hybrid_search
        from src.ingest import get_embeddings_model

        client = get_opensearch_client()
        embeddings = get_embeddings_model()
        query_vector = embeddings.embed_query(query)

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
    """Hybrid retriever: OpenSearch BM25 + k-NN, optionally reranked."""

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
    """Build the hybrid retriever with optional reranking."""
    retriever = HybridRetriever(rerank=rerank, k=10)
    print(f"Hybrid retriever built: OpenSearch BM25 + k-NN" + (" + reranker" if rerank else ""))
    return retriever

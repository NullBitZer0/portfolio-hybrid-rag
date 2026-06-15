from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.config import BM25_WEIGHT, VECTOR_WEIGHT, RETRIEVAL_K, RERANKER_MODEL, RERANK_TOP_K


def build_hybrid_retriever(
    vectorstore: Chroma,
    bm25_retriever: BM25Retriever,
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
    k: int = RETRIEVAL_K,
) -> EnsembleRetriever:
    """Combine BM25 keyword search + vector semantic search using RRF."""
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25_retriever.k = k

    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
    print(f"Hybrid retriever built: BM25({bm25_weight}) + Vector({vector_weight})")
    return hybrid


def build_reranker(top_k: int = RERANK_TOP_K) -> CrossEncoderReranker:
    """Build cross-encoder re-ranker for refining retrieval results."""
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_k)
    print(f"Reranker built: {RERANKER_MODEL} (top {top_k})")
    return reranker


def get_retriever(vectorstore: Chroma, bm25_retriever: BM25Retriever):
    """Returns hybrid retriever without reranking."""
    return build_hybrid_retriever(vectorstore, bm25_retriever)


def get_reranked_retriever(vectorstore: Chroma, bm25_retriever: BM25Retriever):
    """Returns hybrid retriever with cross-encoder re-ranking."""
    from langchain.retrievers import ContextualCompressionRetriever

    hybrid = build_hybrid_retriever(vectorstore, bm25_retriever, k=10)
    reranker = build_reranker()
    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=hybrid
    )

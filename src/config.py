import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_DIM = 768  # Gemini text-embedding-004
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
RETRIEVAL_K = 5
MEMORY_WINDOW = 5

# OpenSearch Configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "opensearch:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "rag-documents")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"


def get_llm(temperature: float = 0.1):
    from langchain_groq import ChatGroq
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=temperature)

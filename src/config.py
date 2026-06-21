import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
LLM_MODEL = "qwen/qwen3-32b"
MAX_LLM_TOKENS = 1024
EMBEDDING_DIM = 768  # Gemini gemini-embedding-2
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
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Docling
DOCLING_URL = os.getenv("DOCLING_URL", "http://docling:5001")

# Folder categories
FOLDERS = ["", "resume", "completed_ml_projects", "inprogress_ml_projects", "certifications", "uni_projects"]
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


def get_llm(temperature: float = 0.1, max_tokens: int = None):
    from langchain_groq import ChatGroq
    kwargs = {"groq_api_key": GROQ_API_KEY, "model_name": LLM_MODEL, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(**kwargs)

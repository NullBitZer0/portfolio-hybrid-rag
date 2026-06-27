import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger("rag.config")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# Primary LLM (Groq)
LLM_MODEL = "openai/gpt-oss-120b"
MAX_LLM_TOKENS = 4096

# Fallback LLM (Gemini)
FALLBACK_LLM_MODEL = "gemini-2.5-flash"

EMBEDDING_DIM = 768  # Gemini gemini-embedding-2
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
PARENT_CHUNK_SIZE = 2000
BM25_WEIGHT = 0.4
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

# Upstash Redis
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

# API Security
API_KEY = os.getenv("API_KEY", "")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))

# Folder categories
FOLDERS = ["", "resume", "completed_ml_projects", "inprogress_ml_projects", "certifications", "uni_projects"]
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


def get_llm(temperature: float = 0.1, max_tokens: int = None):
    """Get Groq LLM with automatic Gemini fallback on failure."""
    from langchain_groq import ChatGroq

    kwargs = {"groq_api_key": GROQ_API_KEY, "model_name": LLM_MODEL, "temperature": temperature, "max_retries": 3}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(**kwargs)


def get_fallback_llm(temperature: float = 0.1, max_tokens: int = None):
    """Get Gemini fallback LLM."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {"google_api_key": GEMINI_API_KEY, "model": FALLBACK_LLM_MODEL, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatGoogleGenerativeAI(**kwargs)


def invoke_llm_with_fallback(
    primary_kwargs: dict, prompt_or_messages, temperature: float = 0.1, max_tokens: int = None
):
    """Try Groq first, fallback to Gemini on rate limit/connection errors."""
    try:
        llm = get_llm(temperature=temperature, max_tokens=max_tokens)
        return llm.invoke(prompt_or_messages)
    except Exception as e:
        err_str = str(e).lower()
        if any(kw in err_str for kw in ["rate_limit", "429", "503", "connection", "timeout", "overloaded"]):
            logger.warning("Groq unavailable (%s), falling back to Gemini", e)
            try:
                fallback = get_fallback_llm(temperature=temperature, max_tokens=max_tokens)
                return fallback.invoke(prompt_or_messages)
            except Exception as e2:
                logger.error("Gemini fallback also failed: %s", e2)
                raise
        raise

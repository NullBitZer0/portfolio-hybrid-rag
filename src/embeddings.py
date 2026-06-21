import google.generativeai as genai
from src.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "models/text-embedding-004"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Gemini text-embedding-004."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"]


def embed_query(text: str) -> list[float]:
    """Embed a single query using Gemini text-embedding-004."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]

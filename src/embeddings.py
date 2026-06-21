from google import genai
from src.config import GEMINI_API_KEY, EMBEDDING_DIM

client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "models/gemini-embedding-2"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Gemini gemini-embedding-2."""
    results = []
    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[text],
            config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": EMBEDDING_DIM},
        )
        results.append(result.embeddings[0].values)
    return results


def embed_query(text: str) -> list[float]:
    """Embed a single query using Gemini gemini-embedding-2."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": EMBEDDING_DIM},
    )
    return result.embeddings[0].values

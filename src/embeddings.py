from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from src.config import GEMINI_API_KEY, EMBEDDING_DIM

client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "models/gemini-embedding-2"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _embed_single(text: str, task_type: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config={"task_type": task_type, "output_dimensionality": EMBEDDING_DIM},
    )
    return result.embeddings[0].values


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Gemini gemini-embedding-2."""
    return [_embed_single(text, "RETRIEVAL_DOCUMENT") for text in texts]


def embed_query(text: str) -> list[float]:
    """Embed a single query using Gemini gemini-embedding-2."""
    if not text or len(text.strip()) < 2:
        text = "placeholder"
    return _embed_single(text, "RETRIEVAL_QUERY")

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
RETRIEVAL_K = 5
RERANK_TOP_K = 3
MEMORY_WINDOW = 5

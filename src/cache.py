import hashlib
import json
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage


class LLMCache:
    """Simple in-memory cache for LLM responses."""

    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, dict] = {}
        self.max_size = max_size

    def _make_key(self, query: str, context: str = "") -> str:
        content = f"{query}|{context[:1000]}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, context: str = "") -> Optional[dict]:
        key = self._make_key(query, context)
        if key in self.cache:
            return self.cache[key]
        return None

    def set(self, query: str, response: dict, context: str = ""):
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        key = self._make_key(query, context)
        self.cache[key] = response

    def clear(self):
        self.cache.clear()


llm_cache = LLMCache()

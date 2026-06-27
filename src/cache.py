import hashlib
import json
import logging
from typing import Optional

from src.config import UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN

logger = logging.getLogger("rag.cache")


class LLMCache:
    """Redis-backed cache for LLM responses with in-memory fallback."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._redis = None
        self._fallback: dict[str, dict] = {}

        if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
            try:
                from upstash_redis import Redis

                self._redis = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)
                self._redis.ping()
                logger.info("Redis cache connected (Upstash)")
            except Exception as e:
                logger.warning("Redis unavailable, using in-memory cache: %s", e)
                self._redis = None
        else:
            logger.info("No Redis config, using in-memory cache")

    def _make_key(self, query: str, context: str = "") -> str:
        content = f"{query}|{context[:1000]}"
        return f"rag:cache:{hashlib.md5(content.encode()).hexdigest()}"

    def get(self, query: str, context: str = "") -> Optional[dict]:
        key = self._make_key(query, context)

        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    return json.loads(data)
                return None
            except Exception:
                pass

        return self._fallback.get(key)

    def set(self, query: str, response: dict, context: str = ""):
        key = self._make_key(query, context)

        if self._redis:
            try:
                self._redis.setex(key, self.ttl, json.dumps(response))
                return
            except Exception:
                pass

        # Fallback: in-memory with max_size eviction
        if len(self._fallback) >= self.max_size:
            oldest_key = next(iter(self._fallback))
            del self._fallback[oldest_key]
        self._fallback[key] = response

    def clear(self):
        if self._redis:
            try:
                keys = self._redis.keys("rag:cache:*")
                if keys:
                    self._redis.delete(*keys)
            except Exception:
                pass
        self._fallback.clear()


llm_cache = LLMCache()

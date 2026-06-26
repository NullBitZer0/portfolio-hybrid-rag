import json
from langchain_core.messages import HumanMessage, AIMessage
from src.config import MEMORY_WINDOW, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN


class ConversationMemory:
    """Redis-backed conversation memory with in-memory fallback."""

    def __init__(self, session_id: str = "default", max_window: int = MEMORY_WINDOW):
        self.session_id = session_id
        self.max_window = max_window
        self._redis = None
        self._fallback: list[HumanMessage | AIMessage] = []
        self._key = f"rag:memory:{session_id}"
        self._ttl = 86400  # 24 hours

        if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
            try:
                from upstash_redis import Redis
                self._redis = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)
                self._redis.ping()
                print(f"Redis memory connected (session: {session_id})")
            except Exception as e:
                print(f"Redis unavailable for memory, using in-memory: {e}")
                self._redis = None

    def _load_history(self) -> list[HumanMessage | AIMessage]:
        if self._redis:
            try:
                data = self._redis.get(self._key)
                if data:
                    messages = json.loads(data)
                    return [
                        HumanMessage(content=m["content"]) if m["type"] == "human"
                        else AIMessage(content=m["content"])
                        for m in messages
                    ]
            except Exception:
                pass
        return list(self._fallback)

    def _save_history(self, history: list[HumanMessage | AIMessage]):
        if self._redis:
            try:
                messages = [
                    {"type": "human", "content": m.content} if isinstance(m, HumanMessage)
                    else {"type": "ai", "content": m.content}
                    for m in history
                ]
                self._redis.setex(self._key, self._ttl, json.dumps(messages))
                return
            except Exception:
                pass
        self._fallback = history

    def add_user_message(self, content: str):
        history = self._load_history()
        history.append(HumanMessage(content=content))
        # Trim to max_window (keep pairs)
        if len(history) > self.max_window * 2:
            history = history[-(self.max_window * 2):]
        self._save_history(history)

    def add_ai_message(self, content: str):
        history = self._load_history()
        history.append(AIMessage(content=content))
        if len(history) > self.max_window * 2:
            history = history[-(self.max_window * 2):]
        self._save_history(history)

    def get_history(self) -> list[HumanMessage | AIMessage]:
        return self._load_history()

    def clear(self):
        if self._redis:
            try:
                self._redis.delete(self._key)
            except Exception:
                pass
        self._fallback.clear()


def build_conversational_rag_chain(graph, memory: ConversationMemory):
    def invoke(question: str, source_filter: str = None) -> dict:
        initial_state = {
            "question": question,
            "cleaned": "",
            "messages": [HumanMessage(content=question)],
            "answer": "",
            "tools_used": [],
            "retrieval_round": 0,
            "max_rounds": 2,
            "blocked": False,
            "flags": [],
            "sources": [],
            "pages": [],
            "confidence": 0.0,
        }

        result = graph.invoke(initial_state)

        memory.add_user_message(question)
        memory.add_ai_message(result.get("answer", ""))

        return result

    return invoke

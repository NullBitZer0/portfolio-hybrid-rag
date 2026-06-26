from langchain_core.messages import HumanMessage, AIMessage
from src.config import MEMORY_WINDOW


class ConversationMemory:
    def __init__(self, max_window: int = MEMORY_WINDOW):
        self.history: list[HumanMessage | AIMessage] = []
        self.max_window = max_window

    def add_user_message(self, content: str):
        self.history.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.history.append(AIMessage(content=content))

    def clear(self):
        self.history = []


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

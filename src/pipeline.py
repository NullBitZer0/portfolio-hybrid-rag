from langchain_core.messages import HumanMessage, AIMessage
from src.config import MEMORY_WINDOW
from src.graph import rag_pipeline


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


def build_conversational_rag_chain(retriever, memory: ConversationMemory):
    def invoke(question: str) -> dict:
        result = rag_pipeline(question, retriever)

        memory.add_user_message(question)
        memory.add_ai_message(result.get("answer", ""))

        return result

    return invoke

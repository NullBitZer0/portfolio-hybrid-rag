from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from src.config import GROQ_API_KEY, LLM_MODEL, MEMORY_WINDOW

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions based on the provided context.
Always cite the source when possible. If the context doesn't contain enough information,
say "I don't have enough information to answer that question."
Be concise and direct.

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs) -> str:
    """Format retrieved documents into a single string."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def format_chat_history(history: list) -> str:
    """Format chat history list into readable string."""
    if not history:
        return "No previous conversation."
    formatted = []
    for msg in history[-MEMORY_WINDOW:]:
        role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


class ConversationMemory:
    """Manages multi-turn conversation history."""

    def __init__(self, max_window: int = MEMORY_WINDOW):
        self.history: list[HumanMessage | AIMessage] = []
        self.max_window = max_window

    def add_user_message(self, content: str):
        self.history.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.history.append(AIMessage(content=content))

    def get_history(self) -> list:
        return self.history[-self.max_window:]

    def clear(self):
        self.history = []


def build_rag_chain(retriever):
    """Build the full RAG chain: retrieve -> format -> prompt -> generate."""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.1,
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: format_chat_history([]),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


def build_conversational_rag_chain(retriever, memory: ConversationMemory):
    """Build conversational RAG chain with memory."""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.1,
    )

    def invoke(question: str) -> str:
        history = memory.get_history()
        context_docs = retriever.invoke(question)
        context = format_docs(context_docs)
        history_str = format_chat_history(history)

        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": history_str,
        })

        memory.add_user_message(question)
        memory.add_ai_message(answer)
        return answer

    return invoke


def query(rag_chain, question: str) -> str:
    """Run a question through the RAG chain."""
    return rag_chain.invoke(question)

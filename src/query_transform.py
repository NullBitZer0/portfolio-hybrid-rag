from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import GROQ_API_KEY, LLM_MODEL


# --- Query Rewriting ---

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's question to be clearer and more specific for document retrieval.
Keep the original meaning. Output ONLY the rewritten question, nothing else.

Original: {question}
Rewritten:"""
)


def rewrite_query(question: str, llm: ChatGroq) -> str:
    """Rewrite a vague question into a clearer one."""
    chain = REWRITE_PROMPT | llm | (lambda x: x.content.strip())
    return chain.invoke({"question": question})


# --- Multi-Query ---

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI assistant. Generate {n} different versions of the user's question
for document retrieval. Each version should approach the topic from a different angle.
Output ONLY the queries, one per line, no numbering or labels.

Question: {question}"""
)


def multi_query(question: str, llm: ChatGroq, n: int = 3) -> list[str]:
    """Generate multiple phrasings of a question."""
    chain = MULTI_QUERY_PROMPT | llm | (lambda x: x.content.strip())
    result = chain.invoke({"question": question, "n": n})
    queries = [q.strip() for q in result.split("\n") if q.strip()]
    return queries[:n]


# --- Step-Back ---

STEPBACK_PROMPT = ChatPromptTemplate.from_template(
    """Given this specific question, generate a more general, broader question
that would provide foundational context. Output ONLY the broader question.

Specific: {question}
Broader:"""
)


def step_back_query(question: str, llm: ChatGroq) -> str:
    """Zoom out to a broader concept for better context retrieval."""
    chain = STEPBACK_PROMPT | llm | (lambda x: x.content.strip())
    return chain.invoke({"question": question})


# --- Strategy Router ---

def classify_query(question: str) -> str:
    """Route query to the best transformation strategy."""
    lower = question.lower()
    words = len(question.split())

    ambiguous_indicators = ["tell me about", "what is", "explain", "describe", "info on"]
    specific_indicators = ["how to", "step by step", "code for", "implement", "debug"]
    vague_indicators = ["stuff", "things", "something", "anything", "related"]

    if any(ind in lower for ind in vague_indicators) or words < 4:
        return "multi_query"
    if any(ind in lower for ind in specific_indicators):
        return "step_back"
    if any(ind in lower for ind in ambiguous_indicators):
        return "rewrite"
    return "direct"


def transform_query(question: str, llm: ChatGroq = None) -> dict:
    """Full query transformation pipeline: classify + transform."""
    if llm is None:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.3)

    strategy = classify_query(question)

    result = {
        "original": question,
        "cleaned": question,
        "strategy": strategy,
        "queries": [question],
    }

    if strategy == "multi_query":
        result["queries"] = [question] + multi_query(question, llm, n=2)
    elif strategy == "rewrite":
        result["queries"] = [rewrite_query(question, llm)]
    elif strategy == "step_back":
        result["queries"] = [step_back_query(question, llm), question]

    return result

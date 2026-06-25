from src.config import get_llm


def generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer to the query for better embedding.

    Skips HyDE for very short queries (< 10 chars) since they work fine with BM25.
    """
    if len(query.strip()) < 10:
        return query

    llm = get_llm(temperature=0.3, max_tokens=200)
    prompt = f"""Write a short, specific answer about Adeesha Perera's portfolio.
Write as if you already know the answer. Be factual and concise.
Do not write generic definitions. Be specific about Adeesha's work.

Question: {query}

Answer:"""
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        if len(answer) < 10:
            return query
        return answer
    except Exception:
        return query

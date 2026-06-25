from src.config import get_llm


def generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer to the query for better embedding."""
    llm = get_llm(temperature=0.3, max_tokens=200)
    prompt = f"""Write a short, specific answer to this question. 
Write as if you already know the answer. Be factual and concise.
Do not say "I don't know" or hedge. Just write a direct answer.

Question: {query}

Answer:"""
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        if len(answer) < 5:
            return query
        return answer
    except Exception:
        return query

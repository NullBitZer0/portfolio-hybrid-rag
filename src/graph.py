from src.config import get_llm, MAX_LLM_TOKENS
from src.guardrails import guard_input, guard_toxicity, classify_question, REFUSAL_RESPONSE
from src.query_transform import transform_query, classify_query
from src.utils import format_docs
from src.langfuse_integration import trace


def rag_pipeline(question: str, retriever) -> dict:

    with trace("input_guard", {"question": question}) as span:
        result = guard_input(question)
        span.update(output={"passed": result.passed, "reason": result.reason})
        if not result.passed:
            return {
                "answer": result.reason,
                "confidence": 0.0,
                "mode": "hybrid",
                "grounded": False,
                "strategy": "",
                "blocked": True,
                "flags": [],
            }
        cleaned = result.cleaned

    with trace("classify", {"query": cleaned}) as span:
        classification = classify_question(cleaned)
        span.update(output=classification)
        if not classification["is_portfolio"]:
            return {
                "answer": REFUSAL_RESPONSE,
                "confidence": 0.0,
                "mode": "hybrid",
                "grounded": False,
                "strategy": "",
                "blocked": True,
                "flags": ["general_knowledge"],
            }

    with trace("transform", {"query": cleaned}) as span:
        strategy = classify_query(cleaned)
        if strategy != "direct":
            llm = get_llm(temperature=0.3)
            transform_result = transform_query(cleaned, llm)
            cleaned = transform_result["queries"][0]
            span.update(output={"strategy": strategy, "queries": transform_result["queries"]})
        else:
            span.update(output={"strategy": "direct", "skipped": True})

    with trace("retrieve", {"query": cleaned}) as span:
        docs = retriever.invoke(cleaned)
        context = format_docs(docs)
        span.update(output={"docs_count": len(docs)})

    from src.cache import llm_cache
    cached = llm_cache.get(cleaned, context)
    if cached:
        with trace("generate", {"query": cleaned, "cached": True}) as span:
            span.update(output={"answer": cached["answer"][:200], "cached": True})
            answer = cached["answer"]
    else:
        llm = get_llm(max_tokens=MAX_LLM_TOKENS)
        with trace("generate", {"query": cleaned}) as span:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_template(
                """You are Adeesha's portfolio assistant.

Answer only using retrieved documents. Be concise - max 3-4 sentences.

If the question is unrelated to Adeesha, his projects,
skills, experience, resume, or education, politely refuse.

Do not answer general knowledge questions.
Do not write code unrelated to the portfolio.

Context:
{context}

Question: {question}

Answer:"""
            )
            chain = prompt | llm | (lambda x: x.content.strip())
            answer = chain.invoke({"context": context, "question": cleaned})
            llm_cache.set(cleaned, {"answer": answer}, context)
            span.update(output={"answer": answer[:200]})

    with trace("output_guard", {"answer": answer[:100]}) as span:
        toxicity = guard_toxicity(answer)
        if not toxicity.passed:
            span.update(output={"blocked": True, "reason": toxicity.reason})
            answer = toxicity.fixed

    return {
        "answer": answer,
        "confidence": 1.0,
        "mode": "hybrid",
        "grounded": True,
        "strategy": strategy,
        "blocked": False,
        "flags": [],
    }

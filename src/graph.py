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
        sources = list(set(doc.metadata.get("source", "") for doc in docs))
        pages = list(set(str(doc.metadata.get("page", "")) for doc in docs if doc.metadata.get("source")))
        span.update(output={"docs_count": len(docs), "sources": sources})

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
                """You are Adeesha's portfolio assistant. Answer using ONLY the context below.

RULES:
- When asked about projects, mention what you find in the context. If context shows university or learning projects, describe those.
- Extract specific facts, names, technologies, metrics from the context
- If asked about skills, describe what you find in the context (technical skills, soft skills, or both)
- Never describe the RAG system itself - answer about Adeesha's work
- If the context truly has zero relevant information, say "I don't have that information"
- Be specific and factual - no vague meta-descriptions
- NEVER use markdown formatting. No bullet points, no bold, no headers. Write plain text only.
- Write in short, natural sentences. 2-3 sentences max per answer.
- Do not use special characters like *, #, -, or > in your response.

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
        "sources": sources,
        "pages": pages,
    }

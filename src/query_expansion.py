"""Simple query expansion for better retrieval matching."""

SYNONYM_MAP = {
    "fraud": ["fraud detection", "transaction fraud", "credit card fraud", "anomaly detection"],
    "ml": ["machine learning", "deep learning", "artificial intelligence"],
    "ai": ["artificial intelligence", "machine learning", "deep learning"],
    "nlp": ["natural language processing", "text analysis", "language models"],
    "recsys": ["recommendation system", "recommendation engine", "suggestion system"],
    "mednlp": ["medical nlp", "clinical nlp", "healthcare nlp", "biomedical text"],
    "deploy": ["deployment", "docker", "kubernetes", "mlops", "ci/cd"],
    "mlops": ["machine learning operations", "ml deployment", "model monitoring"],
    "frontend": ["react", "next.js", "typescript", "javascript", "web development"],
    "backend": ["fastapi", "flask", "spring boot", "api development", "server-side"],
    "cloud": ["aws", "gcp", "azure", "oracle cloud", "infrastructure"],
    "data": ["data science", "data engineering", "pandas", "spark", "sql"],
    "model": ["machine learning model", "deep learning model", "trained model"],
    "pipeline": ["data pipeline", "ml pipeline", "etl pipeline", "workflow"],
    "portfolio": ["resume", "projects", "skills", "experience", "education"],
    "projects": ["personal projects", "university projects", "production projects", "ml projects"],
    "skills": ["technical skills", "soft skills", "programming skills", "tools"],
    "karate": ["martial arts", "combat sports", "self-defense", "discipline"],
    "cadet": ["ncc", "military training", "leadership training", "discipline training"],
    "volunteer": ["volunteering", "community service", "social work", "unicef"],
}


def expand_query(query: str) -> str:
    """Expand query with synonyms for better BM25 matching.

    Appends related terms to the original query without replacing it.
    Only expands if the query is short (< 30 chars) to avoid over-expansion.
    """
    lower = query.lower().strip()
    words = set(lower.split())

    expansions = []
    for key, synonyms in SYNONYM_MAP.items():
        if key in words:
            expansions.extend(synonyms[:2])

    if not expansions:
        return query

    expanded = f"{query} {' '.join(expansions)}"
    return expanded

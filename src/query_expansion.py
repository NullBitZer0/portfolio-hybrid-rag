"""Simple query expansion for better retrieval matching.

Expands short queries with related terms to improve BM25 recall.
Only expands standalone words, not words part of compound terms.
"""

SYNONYM_MAP = {
    "fraud": ["fraud detection", "credit card fraud", "anomaly detection"],
    "ml": ["machine learning", "deep learning"],
    "ai": ["artificial intelligence", "machine learning"],
    "nlp": ["natural language processing", "text analysis"],
    "recsys": ["recommendation system", "recommendation engine"],
    "mednlp": ["medical nlp", "clinical nlp", "biomedical text"],
    "deploy": ["deployment", "docker", "kubernetes", "mlops"],
    "mlops": ["machine learning operations", "ml deployment"],
    "frontend": ["react", "next.js", "typescript", "javascript"],
    "backend": ["fastapi", "flask", "spring boot"],
    "cloud": ["aws", "gcp", "azure", "oracle cloud"],
    "karate": ["martial arts", "combat sports", "discipline"],
    "cadet": ["ncc", "military training", "leadership training"],
    "volunteer": ["volunteering", "community service", "unicef"],
}

# Words too generic to expand (would cause noisy results)
STOP_EXPAND = {"skills", "project", "projects", "experience", "work", "job", "tools", "data", "model"}


def expand_query(query: str) -> str:
    """Expand query with synonyms for better BM25 matching.

    Appends related terms to the original query without replacing it.
    Skips expansion for very long queries (> 40 chars) and generic words.
    """
    if len(query.strip()) > 40:
        return query

    lower = query.lower().strip()
    words = set(lower.split())

    expansions = []
    for key, synonyms in SYNONYM_MAP.items():
        if key in words and key not in STOP_EXPAND:
            expansions.extend(synonyms[:2])

    if not expansions:
        return query

    expanded = f"{query} {' '.join(expansions)}"
    return expanded

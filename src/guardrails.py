import re
from src.config import get_llm


# ============================================================
# INPUT GUARDS
# ============================================================

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"you are now",
    r"act as",
    r"pretend (you|to be)",
    r"disregard",
    r"system prompt",
    r"jailbreak",
    r"DAN mode",
    r"bypass",
    r"override",
]

HARMFUL_PATTERNS = [
    r"how to (hack|exploit|phish|steal|bomb|kill)",
    r"(bypass|circvent) (security|auth|payment)",
    r"generate (malware|virus|ransomware)",
    r"(credit card|ssn|social security) number",
]

PORTFOLIO_KEYWORDS = [
    "adeesha",
    "resume",
    "portfolio",
    "project",
    "experience",
    "skill",
    "education",
    "work",
    "job",
    "career",
    "ml",
    "machine learning",
    "data",
    "engineer",
    "scientist",
    "fraud",
    "mednlp",
    "recsys",
    "nlp",
    "deep learning",
    "python",
    "pytorch",
    "tensorflow",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
    "mlops",
    "deployment",
    "certification",
    "course",
    "coursework",
    "reading",
    "soft skills",
    "technical skills",
    "university",
    "slit",
    "slit",
    "certificate",
    "training",
    "learning",
]


class InputGuardResult:
    def __init__(self, passed: bool, reason: str = "", cleaned: str = ""):
        self.passed = passed
        self.reason = reason
        self.cleaned = cleaned

    def __bool__(self):
        return self.passed


def guard_input(query: str) -> InputGuardResult:
    cleaned = query.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    lower = cleaned.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return InputGuardResult(False, "Blocked: potential prompt injection detected.")

    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, lower):
            return InputGuardResult(False, "Blocked: harmful content detected.")

    if len(cleaned) < 2:
        return InputGuardResult(False, "Query too short.")

    if len(cleaned) > 2000:
        return InputGuardResult(False, "Query too long (max 2000 characters).")

    return InputGuardResult(True, cleaned=cleaned)


def classify_question(query: str) -> dict:
    """Classify if question is about Adeesha's portfolio or general knowledge."""
    lower = query.lower()

    has_portfolio_keyword = any(kw in lower for kw in PORTFOLIO_KEYWORDS)

    if has_portfolio_keyword:
        return {"is_portfolio": True, "reason": "Portfolio-related question"}

    llm = get_llm(temperature=0.0)
    prompt = f"""Classify this question. Is it about Adeesha Perera's portfolio, resume, projects, skills, experience, or education?

Question: {query}

Answer ONLY "YES" if it's about Adeesha's portfolio, or "NO" if it's general knowledge."""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip().upper()
        is_portfolio = "YES" in answer
        return {
            "is_portfolio": is_portfolio,
            "reason": "Portfolio-related" if is_portfolio else "General knowledge - not about Adeesha",
        }
    except Exception:
        return {"is_portfolio": True, "reason": "Classification failed, allowing"}


REFUSAL_RESPONSE = "I'm Adeesha's portfolio assistant. I can only answer questions about Adeesha, his projects, skills, experience, resume, or education. Please ask a portfolio-related question."


# ============================================================
# OUTPUT GUARDS
# ============================================================


class OutputGuardResult:
    def __init__(self, passed: bool, reason: str = "", score: float = 1.0, fixed: str = ""):
        self.passed = passed
        self.reason = reason
        self.score = score
        self.fixed = fixed or None

    def __bool__(self):
        return self.passed


def guard_toxicity(answer: str) -> OutputGuardResult:
    toxic_keywords = [
        "stupid",
        "idiot",
        "hate",
        "racist",
        "sexist",
        "slur",
        "kill yourself",
        "die",
        " worthless",
    ]
    lower = answer.lower()
    for kw in toxic_keywords:
        if kw in lower:
            return OutputGuardResult(
                False,
                f"Toxic content detected: '{kw}'",
                score=0.0,
                fixed="I apologize, but I cannot provide that response.",
            )
    return OutputGuardResult(True, score=1.0)

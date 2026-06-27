import logging
from src.config import LANGFUSE_ENABLED, LLM_MODEL, FALLBACK_LLM_MODEL

logger = logging.getLogger("rag.langfuse")

# Model pricing (per 1M tokens) for cost tracking
MODEL_PRICING = {
    LLM_MODEL: {"input": 0.15, "output": 0.60},  # Groq gpt-oss-120b
    FALLBACK_LLM_MODEL: {"input": 0.15, "output": 0.60},  # Gemini 2.5 Flash
}


def trace(name: str, metadata: dict = None):
    """Create a trace span for non-LLM operations."""
    if not LANGFUSE_ENABLED:
        return _DummySpan()
    from langfuse import get_client

    langfuse = get_client()
    return langfuse.start_as_current_observation(
        name=name,
        as_type="span",
        metadata=metadata or {},
    )


def trace_generation(name: str, model: str, input_data, metadata: dict = None):
    """Create a generation span for LLM calls with token tracking."""
    if not LANGFUSE_ENABLED:
        return _DummyGeneration()
    from langfuse import get_client

    langfuse = get_client()
    return langfuse.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        input=input_data,
        metadata=metadata or {},
    )


def track_usage(generation, response, model: str):
    """Extract token usage from LLM response and update generation."""
    if not response:
        return

    usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        usage = {
            "input_tokens": getattr(meta, "input_tokens", 0) or 0,
            "output_tokens": getattr(meta, "output_tokens", 0) or 0,
            "total_tokens": getattr(meta, "total_tokens", 0) or 0,
        }
    elif hasattr(response, "response_metadata") and response.response_metadata:
        rm = response.response_metadata
        if "token_usage" in rm:
            tu = rm["token_usage"]
            usage = {
                "input_tokens": getattr(tu, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(tu, "completion_tokens", 0) or 0,
                "total_tokens": getattr(tu, "total_tokens", 0) or 0,
            }
        elif "usage" in rm:
            u = rm["usage"]
            usage = {
                "input_tokens": u.get("prompt_tokens", 0),
                "output_tokens": u.get("completion_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
            }

    if usage and generation:
        generation.update(
            usage_details=usage,
            output=str(response.content)[:500] if hasattr(response, "content") else "",
        )

    return usage


def calculate_cost(usage: dict, model: str) -> dict:
    """Calculate cost based on token usage and model pricing."""
    if not usage or model not in MODEL_PRICING:
        return {}

    pricing = MODEL_PRICING[model]
    input_cost = (usage.get("input_tokens", 0) / 1_000_000) * pricing["input"]
    output_cost = (usage.get("output_tokens", 0) / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input": round(input_cost, 6),
        "output": round(output_cost, 6),
        "total": round(total_cost, 6),
    }


def add_score(name: str, value: float, comment: str = ""):
    """Add a score to the current trace."""
    if not LANGFUSE_ENABLED:
        return
    from langfuse import get_client

    langfuse = get_client()
    langfuse.score(name=name, value=value, comment=comment)


class _DummySpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, **kwargs):
        pass


class _DummyGeneration:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, **kwargs):
        pass

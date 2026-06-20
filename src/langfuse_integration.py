from src.config import LANGFUSE_ENABLED


def trace(name: str, metadata: dict = None):
    if not LANGFUSE_ENABLED:
        return _DummySpan()
    from langfuse import get_client
    langfuse = get_client()
    return langfuse.start_as_current_observation(
        name=name,
        as_type="span",
        metadata=metadata or {},
    )


class _DummySpan:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, **kwargs):
        pass

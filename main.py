import sys


def cli_mode():
    """Interactive CLI chat with memory."""
    print("=" * 60)
    print("  Agentic RAG - CLI Mode")
    print("  LangGraph Agent | OpenSearch | Groq LLM")
    print("=" * 60)

    print("\n[1/2] Loading agentic graph...")
    from src.graph import rag_graph

    print("[2/2] Starting chat (type 'quit' to exit, 'clear' to reset memory)\n")
    from src.pipeline import build_conversational_rag_chain, ConversationMemory
    memory = ConversationMemory()
    chain = build_conversational_rag_chain(rag_graph, memory)

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "clear":
            memory.clear()
            print("Memory cleared.\n")
            continue
        if not question:
            continue
        result = chain(question)
        print(f"\nAssistant: {result['answer']}")
        print(f"[Confidence: {result['confidence']} | Sources: {result.get('sources', [])}]\n")


def api_mode():
    """Start FastAPI server."""
    import uvicorn
    print("=" * 60)
    print("  Agentic RAG - API Mode")
    print("  http://localhost:8000")
    print("=" * 60)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cli"
    if mode == "api":
        api_mode()
    else:
        cli_mode()

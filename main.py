import sys
from src.ingest import ingest
from src.retrieval import get_reranked_retriever
from src.pipeline import build_conversational_rag_chain, ConversationMemory


def cli_mode():
    """Interactive CLI chat with memory."""
    print("=" * 60)
    print("  Hybrid RAG - CLI Mode")
    print("  BM25 + Vector + Re-ranking | ChromaDB | Groq LLM")
    print("=" * 60)

    print("\n[1/3] Ingesting documents...")
    vectorstore, bm25_retriever, chunks = ingest()

    print("[2/3] Building hybrid retriever + reranker...")
    retriever = get_reranked_retriever(vectorstore, bm25_retriever)

    print("[3/3] Starting chat (type 'quit' to exit, 'clear' to reset memory)\n")
    memory = ConversationMemory()
    chain = build_conversational_rag_chain(retriever, memory)

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
        print(f"[Confidence: {result['confidence']} | Mode: {result['mode']}", end="")
        if result.get("iterations", 1) > 1:
            print(f" | Iterations: {result['iterations']}", end="")
        if result.get("flags"):
            print(f" | Flags: {result['flags']}", end="")
        print("]\n")


def api_mode():
    """Start FastAPI server."""
    import uvicorn
    print("=" * 60)
    print("  Hybrid RAG - API Mode")
    print("  http://localhost:8000")
    print("=" * 60)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cli"
    if mode == "api":
        api_mode()
    else:
        cli_mode()

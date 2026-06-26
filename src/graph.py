from typing import TypedDict, Annotated, Literal
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.config import get_llm, MAX_LLM_TOKENS
from src.guardrails import guard_input, guard_toxicity, classify_question, REFUSAL_RESPONSE
from src.langfuse_integration import trace
from src.tools import search_all, search_projects, search_skills, search_source, list_documents


# ── State ──────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    cleaned: str
    messages: Annotated[list[BaseMessage], add]
    answer: str
    tools_used: list[str]
    retrieval_round: int
    max_rounds: int
    blocked: bool
    flags: list[str]
    sources: list[str]
    pages: list[str]
    confidence: float


# ── Agent Tools ────────────────────────────────────────────

TOOLS = [search_all, search_projects, search_skills, search_source, list_documents]
TOOL_MAP = {t.name: t for t in TOOLS}

AGENT_SYSTEM_PROMPT = """You are Adeesha Perera's portfolio AI assistant. You have search tools to find information about his projects, skills, experience, and education.

Available documents in the knowledge base:
- technical_skills.pdf - Programming languages (Python, SQL, JS, Java, Kotlin), ML/DL frameworks (PyTorch, TensorFlow, XGBoost, CatBoost, Scikit-learn), DevOps tools (Docker, Kubernetes, AWS), data tools (Pandas, NumPy, Spark)
- soft_skills.pdf - Soft skills (leadership, teamwork, resilience), karate, cadet training, UNICEF volunteering
- all_projects.pdf - All projects categorized (production, university, learning, fun)
- realtime_fraud_detection.pdf - Real-time Fraud Detection System details (CatBoost, Feast, Kafka, Airflow)
- hybrid_rag_project.pdf - Hybrid RAG Portfolio Assistant details (OpenSearch, Gemini, Cohere, Docling)

TOOL ROUTING RULES (IMPORTANT):
- "skills", "technical skills", "tools", "technologies", "programming" -> ALWAYS call search_skills first
- "projects", "built", "created", "developed" -> ALWAYS call search_projects first
- "about adeesha", "who is", "resume", "experience" -> call search_all
- If search_skills returns only soft skills info, also call search_source("technical tools", "technical_skills.pdf")
- Use list_documents to see what is available if unsure

RULES:
- Call tools to find information. Do NOT make up facts.
- You may call multiple tools in one turn.
- After seeing results, decide if you have enough information.
- If you need more details, call more tools (max 2 retrieval rounds total).
- When you have enough information, stop calling tools and say READY.
- NEVER describe the RAG system itself. Answer about Adeesha's work.
- NEVER use markdown formatting. Write plain text only.
- Write in short, natural sentences. 2-3 sentences max per answer.
- Do not use special characters like *, #, -, or > in your response.
"""


# ── Graph Nodes ────────────────────────────────────────────

def input_guard(state: RAGState) -> dict:
    """Validate input query for safety."""
    with trace("input_guard", {"question": state["question"]}) as span:
        result = guard_input(state["question"])
        span.update(output={"passed": result.passed, "reason": result.reason})
        return {
            "cleaned": result.cleaned if result.passed else "",
            "blocked": not result.passed,
        }


def classify_portfolio(state: RAGState) -> dict:
    """Classify if question is about Adeesha's portfolio."""
    with trace("classify", {"query": state["cleaned"]}) as span:
        classification = classify_question(state["cleaned"])
        span.update(output=classification)
        if not classification["is_portfolio"]:
            return {"blocked": True, "flags": ["general_knowledge"]}
        return {}


def agent_step(state: RAGState) -> dict:
    """Agent decides which tools to call or generates final answer."""
    with trace("agent_step", {"round": state["retrieval_round"], "question": state["cleaned"]}) as span:
        llm = get_llm(temperature=0.1, max_tokens=MAX_LLM_TOKENS)
        llm_with_tools = llm.bind_tools(TOOLS)

        # Build messages: system + user query + previous tool results
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

        # Add conversation context
        for msg in state["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                messages.append(msg)

        response = llm_with_tools.invoke(messages)

        tools_used = []
        if response.tool_calls:
            tools_used = [tc["name"] for tc in response.tool_calls]

        print(f"Agent step (round {state['retrieval_round']}): tools={tools_used}")

        span.update(output={
            "tool_calls": tools_used,
            "has_tool_calls": bool(response.tool_calls),
        })

        return {
            "messages": [response],
            "tools_used": tools_used,
        }


def analyze_results(state: RAGState) -> dict:
    """Analyze retrieved results and decide if more retrieval is needed."""
    with trace("analyze", {"round": state["retrieval_round"]}) as span:
        # Check if we hit max rounds
        if state["retrieval_round"] >= state["max_rounds"]:
            span.update(output={"decision": "enough", "reason": "max_rounds"})
            return {}

        # Ask LLM if we have enough info
        llm = get_llm(temperature=0.0, max_tokens=100)

        # Collect all tool results from messages
        tool_results = []
        for msg in state["messages"]:
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.type == "tool":
                tool_results.append(msg.content[:500])

        context_summary = "\n\n".join(tool_results) if tool_results else "No results yet"

        prompt = f"""Based on these search results, do you have enough information to answer this question?

Question: {state["cleaned"]}

Search results summary:
{context_summary[:2000]}

Reply with exactly one word: "enough" if you have sufficient information, or "need_more" if you need to search for more details."""

        try:
            response = llm.invoke(prompt)
            decision = response.content.strip().lower()
            needs_more = "need_more" in decision
        except Exception:
            needs_more = False

        span.update(output={"decision": "need_more" if needs_more else "enough"})

        return {"retrieval_round": state["retrieval_round"] + 1}


def generate(state: RAGState) -> dict:
    """Generate final answer from accumulated context."""
    with trace("generate", {"question": state["cleaned"]}) as span:
        # If blocked, return the blocked response
        if state.get("blocked"):
            if state.get("flags") and "general_knowledge" in state["flags"]:
                answer = REFUSAL_RESPONSE
            else:
                answer = guard_input(state["question"]).reason or "Query blocked."
            return {
                "answer": answer,
                "confidence": 0.0,
                "sources": [],
                "pages": [],
            }

        # Collect all tool results
        context_parts = []
        all_sources = set()
        all_pages = set()
        for msg in state["messages"]:
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.type == "tool":
                context_parts.append(msg.content)
                # Extract sources from tool results
                for line in msg.content.split("\n"):
                    if line.startswith("[Source:"):
                        src = line.split("Source:")[1].split(",")[0].strip()
                        page = line.split("Page")[1].split("]")[0].strip() if "Page" in line else ""
                        all_sources.add(src)
                        if page:
                            all_pages.add(page)

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        # Generate answer
        llm = get_llm(temperature=0.1, max_tokens=MAX_LLM_TOKENS)
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            """You are Adeesha's portfolio assistant. Answer using ONLY the context below.

RULES:
- When asked about projects, mention what you find in the context.
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
        answer = chain.invoke({"context": context, "question": state["cleaned"]})

        span.update(output={"answer": answer[:200], "sources": list(all_sources)})

        return {
            "answer": answer,
            "confidence": 1.0,
            "sources": list(all_sources),
            "pages": list(all_pages),
        }


def output_guard(state: RAGState) -> dict:
    """Check output for toxicity."""
    with trace("output_guard", {"answer": state.get("answer", "")[:100]}) as span:
        toxicity = guard_toxicity(state.get("answer", ""))
        if not toxicity.passed:
            span.update(output={"blocked": True, "reason": toxicity.reason})
            return {"answer": toxicity.fixed}
        return {}


# ── Routing Functions ──────────────────────────────────────

def route_after_input_guard(state: RAGState) -> Literal["blocked", "classify"]:
    if state.get("blocked"):
        return "blocked"
    return "classify"


def route_after_classify(state: RAGState) -> Literal["refusal", "agent"]:
    if state.get("blocked"):
        return "refusal"
    return "agent"


def route_after_agent(state: RAGState) -> Literal["tools", "generate"]:
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "generate"


def route_after_analyze(state: RAGState) -> Literal["more", "generate"]:
    if state["retrieval_round"] < state["max_rounds"]:
        # Check if agent actually called tools last round
        last_msg = state["messages"][-1] if state["messages"] else None
        if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "more"
    return "generate"


def tool_executor(state: RAGState) -> dict:
    """Execute tool calls and log results."""
    last_msg = state["messages"][-1] if state["messages"] else None
    if not last_msg or not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"messages": []}

    results = []
    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        print(f"  Tool call: {tool_name}({tool_args})")
        try:
            tool = TOOL_MAP[tool_name]
            result = tool.invoke(tool_args)
            result_preview = str(result)[:150]
            print(f"  Tool result: {result_preview}...")
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        except Exception as e:
            print(f"  Tool error: {e}")
            results.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))

    return {"messages": results}


# ── Build Graph ────────────────────────────────────────────

def build_graph():
    """Build and compile the agentic RAG graph."""
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("input_guard", input_guard)
    graph.add_node("classify_portfolio", classify_portfolio)
    graph.add_node("agent_step", agent_step)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("analyze_results", analyze_results)
    graph.add_node("generate", generate)
    graph.add_node("output_guard", output_guard)

    # Set entry point
    graph.set_entry_point("input_guard")

    # Add edges
    graph.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {"blocked": "generate", "classify": "classify_portfolio"},
    )

    graph.add_conditional_edges(
        "classify_portfolio",
        route_after_classify,
        {"refusal": "generate", "agent": "agent_step"},
    )

    graph.add_conditional_edges(
        "agent_step",
        route_after_agent,
        {"tools": "tool_executor", "generate": "generate"},
    )

    graph.add_edge("tool_executor", "analyze_results")

    graph.add_conditional_edges(
        "analyze_results",
        route_after_analyze,
        {"more": "agent_step", "generate": "generate"},
    )

    graph.add_edge("generate", "output_guard")
    graph.add_edge("output_guard", END)

    return graph.compile()


# ── Compiled Graph (importable) ────────────────────────────

rag_graph = build_graph()

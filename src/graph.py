import logging
from typing import TypedDict, Annotated, Literal
from operator import add

logger = logging.getLogger("rag.graph")

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from src.config import get_llm, invoke_llm_with_fallback, MAX_LLM_TOKENS
from src.guardrails import guard_input, guard_toxicity, classify_question, REFUSAL_RESPONSE
from src.langfuse_integration import trace
from src.tools import search_all, search_projects, search_skills, search_source, list_documents


# ── Prompt Versioning ───────────────────────────────────────

PROMPT_VERSION = "v2.0"


# ── State ──────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    cleaned: str
    source_filter: str
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

AGENT_SYSTEM_PROMPT = f"""[Prompt: {PROMPT_VERSION}] You are Adeesha Perera's portfolio AI assistant. You have search tools to find information about his projects, skills, experience, and education.

Available documents in the knowledge base:
- resume/technical_skills.pdf - Programming languages (Python, SQL, JS, Java, Kotlin), ML/DL frameworks (PyTorch, TensorFlow, XGBoost, CatBoost, Scikit-learn), DevOps tools (Docker, Kubernetes, AWS), data tools (Pandas, NumPy, Spark)
- soft_skills.pdf - Soft skills (leadership, teamwork, resilience), karate, cadet training, UNICEF volunteering
- resume/all_projects.pdf - All projects categorized (production, university, learning, fun)
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

GENERATE_PROMPT = f"""[Prompt: {PROMPT_VERSION}] You are Adeesha's portfolio assistant. Answer using ONLY the context below.

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
{{context}}

Question: {{question}}

Answer:"""


# ── Graph Nodes ────────────────────────────────────────────

def input_guard(state: RAGState) -> dict:
    """Validate input query for safety."""
    with trace("input_guard", {"question": state.get("question", ""), "prompt_version": PROMPT_VERSION}) as span:
        result = guard_input(state.get("question", ""))
        span.update(output={"passed": result.passed, "reason": result.reason})
        return {
            "cleaned": result.cleaned if result.passed else "",
            "blocked": not result.passed,
        }


def classify_portfolio(state: RAGState) -> dict:
    """Classify if question is about Adeesha's portfolio."""
    with trace("classify", {"query": state.get("cleaned", "")}) as span:
        classification = classify_question(state.get("cleaned", ""))
        span.update(output=classification)
        if not classification["is_portfolio"]:
            return {"blocked": True, "flags": ["general_knowledge"]}
        return {}


def agent_step(state: RAGState) -> dict:
    """Agent decides which tools to call or generates final answer."""
    retrieval_round = state.get("retrieval_round", 0)
    source_filter = state.get("source_filter", "")
    with trace("agent_step", {"round": retrieval_round, "question": state.get("cleaned", ""), "prompt_version": PROMPT_VERSION}) as span:
        llm = get_llm(temperature=0.1, max_tokens=MAX_LLM_TOKENS)
        llm_with_tools = llm.bind_tools(TOOLS)

        # Build messages: system + user query + previous tool results
        system_prompt = AGENT_SYSTEM_PROMPT
        if source_filter:
            system_prompt += f"\n\nIMPORTANT: The user wants results from '{source_filter}' only. Use search_source with source='{source_filter}' for all searches."

        messages = [SystemMessage(content=system_prompt)]

        # Add conversation context
        for msg in state.get("messages", []):
            if isinstance(msg, (HumanMessage, AIMessage)):
                messages.append(msg)

        response = llm_with_tools.invoke(messages)

        tools_used = []
        if response.tool_calls:
            tools_used = [tc["name"] for tc in response.tool_calls]

        logger.info("Agent step (round %d): tools=%s", retrieval_round, tools_used)

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
    retrieval_round = state.get("retrieval_round", 0)
    with trace("analyze", {"round": retrieval_round}) as span:
        span.update(output={"decision": "increment_round"})
        return {"retrieval_round": retrieval_round + 1}


def generate(state: RAGState) -> dict:
    """Generate final answer from accumulated context."""
    with trace("generate", {"question": state.get("cleaned", ""), "prompt_version": PROMPT_VERSION}) as span:
        # If blocked, return the blocked response
        if state.get("blocked"):
            if state.get("flags") and "general_knowledge" in state["flags"]:
                answer = REFUSAL_RESPONSE
            else:
                answer = guard_input(state.get("question", "")).reason or "Query blocked."
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
        for msg in state.get("messages", []):
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

        # Generate answer with fallback
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(GENERATE_PROMPT)
        question = state.get("cleaned", "")
        formatted = prompt.format(context=context, question=question)

        try:
            response = invoke_llm_with_fallback({}, formatted, temperature=0.1, max_tokens=MAX_LLM_TOKENS)
            answer = response.content.strip()
        except Exception as e:
            logger.error("All LLM providers failed for generate: %s", e)
            answer = "I'm having trouble generating a response right now. Please try again."

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
    last_msg = state["messages"][-1] if state.get("messages") else None
    if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "generate"


def route_after_analyze(state: RAGState) -> Literal["more", "generate"]:
    if state.get("retrieval_round", 0) < state.get("max_rounds", 2):
        # Check if agent actually called tools last round
        last_msg = state["messages"][-1] if state.get("messages") else None
        if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "more"
    return "generate"


def tool_executor(state: RAGState) -> dict:
    """Execute tool calls and log results."""
    last_msg = state["messages"][-1] if state.get("messages") else None
    if not last_msg or not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"messages": []}

    results = []
    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        logger.info("  Tool call: %s(%s)", tool_name, tool_args)
        try:
            tool = TOOL_MAP[tool_name]
            result = tool.invoke(tool_args)
            result_preview = str(result)[:150]
            logger.info("  Tool result: %s...", result_preview)
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        except Exception as e:
            logger.error("  Tool error: %s", e)
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

"""
Level 2B: Solver-Critic (Dynamic Self-Correction Loop).

Graph:
  solver --> critic --> (conditional edge)
                          |-- "retry" --> solver   (when critic says needs_fix AND cap not hit)
                          |-- "end"   --> END       (when critic says correct OR cap reached)

- Solver (Groq/Llama): generates a candidate solution, incorporating any prior critic feedback.
- Critic (Gemini Flash): audits the solution, returns a structured CriticReview.
- Routing function: reads CriticReview.needs_retry and the current iteration count,
  enforces the iteration_cap passed at build time.

The iteration_cap is the primary hyperparameter sweep variable (caps of 1, 2, 3).
"""

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.llm_client import call_llm, SOLVER_ROLE, CRITIC_ROLE
from src.prompts import solver_prompt, critic_prompt
from src.schemas import CriticReview
from src.tools import retrieve


def _make_solver_node(top_k: int):
    def _solver_node(state: AgentState) -> dict:
        domain = state["domain"]
        documents = state.get("documents", {})
        critic_reviews = state.get("critic_reviews", [])

        context: dict[str, str] | None = None
        if domain == "hotpotqa" and documents:
            query = state["question"]
            if critic_reviews:
                last_errors = " ".join(critic_reviews[-1].errors_found)
                query = query + " " + last_errors
            context = retrieve(query=query, documents=documents, top_k=top_k)
        elif domain == "math":
            context = None

        critic_feedback = None
        previous_solution = None
        if critic_reviews:
            last_review = critic_reviews[-1]
            feedback_parts = []
            if last_review.errors_found:
                feedback_parts.append("Errors found: " + "; ".join(last_review.errors_found))
            if last_review.fix_instructions:
                feedback_parts.append("How to fix: " + last_review.fix_instructions)
            critic_feedback = "\n".join(feedback_parts) if feedback_parts else None
            previous_solution = state.get("current_solution", "")

        lessons = state.get("retrieved_lessons", []) or None

        prompt = solver_prompt(
            question=state["question"],
            domain=domain,
            context=context,
            critic_feedback=critic_feedback,
            previous_solution=previous_solution,
            lessons=lessons if lessons else None,
        )

        result, usage = call_llm(prompt=prompt, role=SOLVER_ROLE)

        return {
            "current_solution": result,
            "total_input_tokens": state.get("total_input_tokens", 0) + usage["input_tokens"],
            "total_output_tokens": state.get("total_output_tokens", 0) + usage["output_tokens"],
            "total_steps": state.get("total_steps", 0) + 1,
        }
    return _solver_node


def _make_critic_node(top_k: int):
    def _critic_node(state: AgentState) -> dict:
        domain = state["domain"]
        documents = state.get("documents", {})

        context: dict[str, str] | None = None
        if domain == "hotpotqa" and documents:
            context = retrieve(
                query=state["question"],
                documents=documents,
                top_k=top_k,
            )

        prompt = critic_prompt(
            question=state["question"],
            domain=domain,
            solution=state["current_solution"],
            context=context,
        )

        review, usage = call_llm(
            prompt=prompt,
            role=CRITIC_ROLE,
            response_schema=CriticReview,
        )

        return {
            "critic_reviews": [review],
            "iteration": state.get("iteration", 0) + 1,
            "total_input_tokens": state.get("total_input_tokens", 0) + usage["input_tokens"],
            "total_output_tokens": state.get("total_output_tokens", 0) + usage["output_tokens"],
            "total_steps": state.get("total_steps", 0) + 1,
        }
    return _critic_node


def _make_router(iteration_cap: int):
    """
    Returns a routing function that reads the latest CriticReview and the iteration count.
    Routes to "retry" or "end".
    """
    def _route(state: AgentState) -> str:
        reviews = state.get("critic_reviews", [])
        iteration = state.get("iteration", 0)

        if not reviews:
            return "end"

        last_review = reviews[-1]

        if last_review.needs_retry and iteration < iteration_cap:
            return "retry"
        return "end"

    return _route


def _finalize_node(state: AgentState) -> dict:
    """Set final_answer from the last solution produced by the solver."""
    return {"final_answer": state.get("current_solution", "")}


def build_graph(iteration_cap: int = 2, top_k: int = 3):
    """
    Build and compile the Level 2B graph.

    Args:
        iteration_cap: Maximum number of solver-critic loops allowed.
        top_k: Number of paragraphs returned by TF-IDF retrieval for HotpotQA.
               Use top_k=10 to match L1's full-context access.
    """
    graph = StateGraph(AgentState)

    graph.add_node("solver", _make_solver_node(top_k))
    graph.add_node("critic", _make_critic_node(top_k))
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("solver")
    graph.add_edge("solver", "critic")

    router = _make_router(iteration_cap)
    graph.add_conditional_edges(
        "critic",
        router,
        {
            "retry": "solver",
            "end": "finalize",
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()

"""
Level 2A: Planner-Executor (Static Feed-forward).

Graph: planner --> executor --> END

- Planner (Gemini Flash): receives the question, outputs a Plan (list of steps).
- Executor (Groq/Llama): receives the plan + question, executes steps sequentially.
  For HotpotQA, runs a retrieval step first to surface relevant paragraphs.
  For GSM8K, works directly from the plan.
No feedback loop — static, one-pass architecture.
"""

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.llm_client import call_llm, PLANNER_ROLE, EXECUTOR_ROLE
from src.prompts import planner_prompt, executor_prompt
from src.schemas import Plan
from src.tools import retrieve


def _planner_node(state: AgentState) -> dict:
    context = state.get("documents") if state.get("document_ids") else None

    prompt = planner_prompt(
        question=state["question"],
        domain=state["domain"],
        context=context,
    )

    plan, usage = call_llm(prompt=prompt, role=PLANNER_ROLE, response_schema=Plan)

    return {
        "plan_steps": plan.steps,
        "total_input_tokens": state.get("total_input_tokens", 0) + usage["input_tokens"],
        "total_output_tokens": state.get("total_output_tokens", 0) + usage["output_tokens"],
        "total_steps": state.get("total_steps", 0) + 1,
    }


def _make_executor_node(top_k: int):
    def _executor_node(state: AgentState) -> dict:
        domain = state["domain"]
        documents = state.get("documents", {})
        retrieved_paragraphs: dict[str, str] | None = None

        if domain == "hotpotqa" and documents:
            retrieval_query = state["question"] + " " + " ".join(state["plan_steps"])
            retrieved_paragraphs = retrieve(
                query=retrieval_query,
                documents=documents,
                top_k=top_k,
            )

        prompt = executor_prompt(
            question=state["question"],
            domain=domain,
            plan_steps=state["plan_steps"],
            context=documents if not retrieved_paragraphs else None,
            retrieved_paragraphs=retrieved_paragraphs,
        )

        result, usage = call_llm(prompt=prompt, role=EXECUTOR_ROLE)

        return {
            "current_solution": result,
            "final_answer": result,
            "total_input_tokens": state.get("total_input_tokens", 0) + usage["input_tokens"],
            "total_output_tokens": state.get("total_output_tokens", 0) + usage["output_tokens"],
            "total_steps": state.get("total_steps", 0) + 1,
        }
    return _executor_node


def build_graph(top_k: int = 3):
    """Build and compile the Level 2A graph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", _planner_node)
    graph.add_node("executor", _make_executor_node(top_k))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", END)

    return graph.compile()

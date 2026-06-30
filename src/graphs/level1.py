"""
Level 1: Bare Single-Agent Baseline.

One node, one LLM call, straight to END.
No scaffolding, no planning, no JSON constraints — honest floor measurement.
Uses the Groq/Llama model (solver/executor role) for the baseline call.
"""

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.llm_client import call_llm, BASELINE_ROLE
from src.prompts import baseline_prompt


def _baseline_node(state: AgentState) -> dict:
    context = state.get("documents") if state.get("document_ids") else None

    prompt = baseline_prompt(
        question=state["question"],
        domain=state["domain"],
        context=context,
    )

    result, usage = call_llm(prompt=prompt, role=BASELINE_ROLE)

    return {
        "current_solution": result,
        "final_answer": result,
        "total_input_tokens": state.get("total_input_tokens", 0) + usage["input_tokens"],
        "total_output_tokens": state.get("total_output_tokens", 0) + usage["output_tokens"],
        "total_steps": state.get("total_steps", 0) + 1,
    }


def build_graph() -> any:
    """Build and compile the Level 1 graph."""
    graph = StateGraph(AgentState)
    graph.add_node("baseline", _baseline_node)
    graph.set_entry_point("baseline")
    graph.add_edge("baseline", END)
    return graph.compile()

"""
All prompt templates in one place.
Each function returns a fully-rendered string ready to pass to call_llm().
"""

from src.schemas import CriticReview, Plan


# ---------------------------------------------------------------------------
# Domain briefs inserted into every prompt to orient the model
# ---------------------------------------------------------------------------

_DOMAIN_BRIEF = {
    "math": (
        "You are solving a competition mathematics problem. "
        "Show your reasoning step by step. "
        r"End your final answer with \boxed{answer} (e.g. \boxed{42} or \boxed{\frac{3}{4}})."
    ),
    "hotpotqa": (
        "You are answering a multi-hop question using the provided context paragraphs. "
        "Identify the relevant evidence across paragraphs and synthesize a concise answer. "
        "Give only the answer string, no padding."
    ),
}


def _domain_brief(domain: str) -> str:
    return _DOMAIN_BRIEF.get(domain, "")


# ---------------------------------------------------------------------------
# Level 1: Bare single-agent baseline
# ---------------------------------------------------------------------------

def baseline_prompt(question: str, domain: str, context: dict[str, str] | None = None) -> str:
    parts = [_domain_brief(domain), ""]
    if context:
        parts.append("Context paragraphs:")
        for pid, text in context.items():
            parts.append(f"[{pid}] {text}")
        parts.append("")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Level 2A: Planner-Executor
# ---------------------------------------------------------------------------

def planner_prompt(question: str, domain: str, context: dict[str, str] | None = None) -> str:
    parts = [
        _domain_brief(domain),
        "",
        "Your task is to create a clear, ordered plan to solve the following question.",
        "Output a JSON object with a 'steps' list. Each step is a concrete action in one short phrase under 10 words (e.g., 'Extract the price of apples', 'Multiply quantity by price').",
        "Use 3 to 5 steps. Do NOT solve the problem — only plan the steps.",
        "",
    ]
    if context:
        parts.append("Available context paragraphs (paragraph IDs you may search):")
        for pid, text in context.items():
            parts.append(f"[{pid}] {text}")
        parts.append("")
    parts.append(f"Question: {question}")
    return "\n".join(parts)


def executor_prompt(
    question: str,
    domain: str,
    plan_steps: list[str],
    context: dict[str, str] | None = None,
    retrieved_paragraphs: dict[str, str] | None = None,
) -> str:
    parts = [
        _domain_brief(domain),
        "",
        "Execute the following plan step by step to answer the question.",
        "",
        "Plan:",
    ]
    for i, step in enumerate(plan_steps, 1):
        parts.append(f"  {i}. {step}")
    parts.append("")

    if retrieved_paragraphs:
        parts.append("Retrieved evidence paragraphs:")
        for pid, text in retrieved_paragraphs.items():
            parts.append(f"[{pid}] {text}")
        parts.append("")
    elif context:
        parts.append("Context paragraphs:")
        for pid, text in context.items():
            parts.append(f"[{pid}] {text}")
        parts.append("")

    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("Work through each step, then give your final answer.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Level 2B: Solver-Critic
# ---------------------------------------------------------------------------

def solver_prompt(
    question: str,
    domain: str,
    context: dict[str, str] | None = None,
    critic_feedback: str | None = None,
    previous_solution: str | None = None,
    lessons: list[str] | None = None,
) -> str:
    parts = [_domain_brief(domain), ""]

    if lessons:
        parts.append("LESSONS FROM PAST FAILURES (read carefully to avoid repeating these mistakes):")
        for i, lesson in enumerate(lessons, 1):
            parts.append(f"  {i}. {lesson}")
        parts.append("")

    if context:
        parts.append("Context paragraphs:")
        for pid, text in context.items():
            parts.append(f"[{pid}] {text}")
        parts.append("")

    parts.append(f"Question: {question}")
    parts.append("")

    if critic_feedback and previous_solution:
        parts.extend([
            "Your previous attempt:",
            previous_solution,
            "",
            "The critic found the following errors and instructions for correction:",
            critic_feedback,
            "",
            # Repeat format instruction because models drift on retries, and
            # explicitly forbid the "context insufficient" escape hatch that
            # causes the solver to give up instead of committing to an answer.
            "Using the context above, provide a corrected answer. "
            "You MUST commit to a definite answer — do not say the context is insufficient.",
            _domain_brief(domain),
        ])
    else:
        # End with "Answer:" — same minimal cue as the baseline prompt.
        # The domain brief at the top already specifies the exact output format
        # (#### for GSM8K, concise string for HotpotQA). "Solve this problem,
        # showing your full reasoning." contradicts HotpotQA's "no padding"
        # brief and causes verbose paragraph-length outputs that fail EM scoring.
        parts.append("Answer:")

    return "\n".join(parts)


def critic_prompt(
    question: str,
    domain: str,
    solution: str,
    context: dict[str, str] | None = None,
) -> str:
    parts = [
        "You are the CRITIC agent. Your job is to detect factual errors in the solution.",
        "Judge ONLY from the context paragraphs provided below. Do NOT use outside knowledge.",
        "Only set verdict='needs_fix' if you can cite a specific paragraph that contradicts the answer.",
        "If you cannot find contradicting evidence in the context, set verdict='correct'.",
        "",
    ]

    # Domain-specific rules for what counts as a correct answer
    if domain == "hotpotqa":
        parts.extend([
            "HotpotQA rules:",
            "- A short answer ('yes', 'no', or a brief entity/phrase) is COMPLETE and VALID — do not flag brevity.",
            "- Only flag if a paragraph explicitly contradicts the answer.",
            "- If the answer is 'yes' or 'no' and the context does not clearly say the opposite, mark as correct.",
            "",
        ])
    elif domain == "math":
        parts.extend([
            "MATH rules:",
            r"- The answer must be in \boxed{...} format. Flag only if the boxed value is mathematically wrong.",
            "- Showing work is optional — only the boxed answer matters.",
            "",
        ])

    parts.extend([
        "Output a JSON object matching this schema:",
        '  {"verdict": "correct" | "needs_fix", "errors_found": [list of error strings], "fix_instructions": "string or null"}',
        "Keep each string in errors_found to one short sentence (under 20 words). Keep fix_instructions under 30 words.",
        "",
    ])

    if context:
        parts.append("Context paragraphs (use ONLY these to verify the answer):")
        for pid, text in context.items():
            parts.append(f"[{pid}] {text}")
        parts.append("")

    parts.extend([
        f"Question: {question}",
        "",
        "Solution to review:",
        solution,
        "",
        "Review (only flag needs_fix if a paragraph above directly contradicts the answer):",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Level 3: Memory nodes
# ---------------------------------------------------------------------------

def memory_retriever_prompt(question: str, lessons: list[dict]) -> str:
    """Select which cached lessons are relevant to the current question."""
    if not lessons:
        return ""
    lesson_text = "\n".join(
        f"  [{i}] {l['summary']}" for i, l in enumerate(lessons)
    )
    return (
        f"Given the following question:\n{question}\n\n"
        f"Select the indices of past lessons that are relevant to solving it.\n"
        f"Output a JSON list of integer indices (e.g. [0, 2]). Output [] if none apply.\n\n"
        f"Past lessons:\n{lesson_text}"
    )


def memory_builder_prompt(question: str, solution: str, critic_feedback: str, corrected_solution: str) -> str:
    """Summarize the failure and fix into a compact lesson."""
    return (
        f"You are summarizing a failure and its fix into a compact lesson for future reference.\n\n"
        f"Question: {question}\n"
        f"Initial (incorrect) solution: {solution}\n"
        f"Critic feedback: {critic_feedback}\n"
        f"Corrected solution: {corrected_solution}\n\n"
        f"Write a 1-2 sentence lesson that captures the exact error type and how to avoid it. "
        f"Be specific (e.g. 'When a problem mentions X, ignore it — it is a distractor')."
    )

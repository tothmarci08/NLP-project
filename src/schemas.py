"""
Pydantic models for structured LLM outputs.
These are the schemas used by planner, critic, and answer extraction.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Output of the Planner node: an ordered list of reasoning/search steps."""
    steps: list[str] = Field(
        description="Ordered list of steps to solve the problem. Each step is a concrete action."
    )


class CriticReview(BaseModel):
    """Output of the Critic node: a structured verdict on the solver's work."""
    verdict: Literal["correct", "needs_fix"] = Field(
        description="'correct' if the solution is valid, 'needs_fix' if errors were found."
    )
    errors_found: list[str] = Field(
        default_factory=list,
        description="List of specific errors identified (arithmetic slips, logical errors, hallucinations). Empty if verdict is 'correct'."
    )
    fix_instructions: Optional[str] = Field(
        default=None,
        description="Concrete instructions for the solver to fix the identified errors. None if verdict is 'correct'."
    )

    @property
    def needs_retry(self) -> bool:
        return self.verdict == "needs_fix"


class AnswerOutput(BaseModel):
    """Structured answer extraction for cases where we need a clean final answer."""
    answer: str = Field(description="The final answer string.")
    reasoning_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of the reasoning chain (1-2 sentences)."
    )

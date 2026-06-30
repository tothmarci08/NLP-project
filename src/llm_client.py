import os
import json
import time
from typing import Any, Optional, Type
from pydantic import BaseModel
from openai import OpenAI


_client: Optional[OpenAI] = None

ENDPOINT_URL = "http://mobydick.elte-dh.hu:23432/v1"
MODEL = "Qwen/Qwen3.6-27B"
MAX_TOKENS = 2048
REQUEST_TIMEOUT = 90  # seconds — MATH solutions are long; allow up to ~2048 tokens

CRITIC_ROLE = "critic"
PLANNER_ROLE = "planner"
ROUTER_ROLE = "router"
SOLVER_ROLE = "solver"
EXECUTOR_ROLE = "executor"
BASELINE_ROLE = "baseline"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("ELTE_API_KEY")
        if not api_key:
            raise EnvironmentError("ELTE_API_KEY not set")
        _client = OpenAI(
            base_url=ENDPOINT_URL,
            api_key=api_key,
            timeout=REQUEST_TIMEOUT,
        )
    return _client


def _pydantic_to_json_schema(model: Type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to a clean JSON schema for the vLLM endpoint.
    Strips top-level keys that vLLM doesn't need and may reject.
    """
    schema = model.model_json_schema()
    for key in ("$schema", "$defs", "title", "description"):
        schema.pop(key, None)
    return schema


def call_llm(
    prompt: str,
    role: str,
    response_schema: Optional[Type[BaseModel]] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> tuple[Any, dict]:
    """
    Call the ELTE Qwen3.6-27B endpoint.

    Args:
        prompt: The full prompt string.
        role: Role constant (for documentation; all route to the same model).
        response_schema: Optional Pydantic model. When provided, structured JSON
                         output is requested and thinking is forcibly disabled
                         (Qwen3 quirk: thinking + structured output conflict).
        temperature: Sampling temperature (0.0 = deterministic).
        max_retries: Retry attempts with exponential backoff on transient errors.

    Returns:
        (result, usage) where result is a str or parsed Pydantic instance,
        and usage = {"input_tokens": int, "output_tokens": int}.
    """
    for attempt in range(max_retries):
        try:
            return _call(prompt, response_schema, temperature)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def _call(
    prompt: str,
    response_schema: Optional[Type[BaseModel]],
    temperature: float,
) -> tuple[Any, dict]:
    client = _get_client()
    use_structured = response_schema is not None

    kwargs: dict = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "extra_body": {
            # enable_thinking=False is applied to ALL calls, but for two
            # independent reasons that should not be conflated:
            #
            # Job 1 — mandatory for structured-output calls (planner, critic):
            #   The endpoint doc states that when thinking is on, structured
            #   output breaks — the JSON lands in reasoning_content instead of
            #   content, causing json.loads() to fail. This is a hard endpoint
            #   constraint, not a design choice.
            #
            # Job 2 — deliberate design choice for free-text calls
            #   (solver, executor, baseline):
            #   All three architectures use the same model and the same
            #   temperature. Disabling thinking ensures the only variable
            #   between L1/L2A/L2B is the graph structure itself, not how much
            #   internal reasoning the model is doing. Thinking-on for the
            #   solver but not the critic would make the comparison noisy and
            #   harder to defend. Our prompts already elicit explicit
            #   chain-of-thought reasoning in the output text, so internal
            #   thinking adds no scientific value here.
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }

    if use_structured:
        schema = _pydantic_to_json_schema(response_schema)
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__.lower(),
                "schema": schema,
            },
        }

    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content.strip()

    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

    if use_structured:
        data = json.loads(text)
        return response_schema.model_validate(data), usage

    return text, usage

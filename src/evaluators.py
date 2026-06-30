"""
Deterministic, programmatic scoring — no LLM calls.

MATH:    extract \\boxed{} from prediction, normalize, compare to gold.
HotpotQA: normalize then compute Exact Match (EM) and token-level F1.
"""

import re
import string
from collections import Counter


# ---------------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------------

def _extract_math_boxed(text: str) -> str | None:
    """
    Extract the content of the LAST \\boxed{...} in text, handling nested braces.
    Uses the last occurrence so that multi-step solutions are scored on the final answer.
    """
    # Find rightmost \boxed{ occurrence
    idx = text.rfind(r'\boxed{')
    if idx < 0:
        return None
    start = idx + len(r'\boxed{')
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def _strip_latex(s: str) -> str:
    """
    Normalize a LaTeX math string for comparison.
    Implements the subset of normalizations from the MATH benchmark's
    is_equiv pipeline that covers the answer types actually seen in the dataset:
    numbers, fractions, tuples, complex numbers, polynomials.
    """
    s = s.strip()
    # Remove line breaks
    s = s.replace('\n', ' ')
    # Remove invisible spacing commands
    s = re.sub(r'\\[,;!]', '', s)
    s = s.replace('\\ ', ' ')
    # Remove \left / \right (purely visual sizing)
    s = s.replace('\\left', '').replace('\\right', '')
    # Normalize display/text frac variants to \frac
    s = re.sub(r'\\[dt]frac\b', r'\\frac', s)
    # Unwrap \text{...} — keep the content
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    # Remove dollar signs
    s = s.replace('$', '')
    # Remove trailing period (answer sometimes ends with ".")
    s = s.rstrip('.')
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


def _to_float(s: str) -> float | None:
    """
    Try to interpret a normalized LaTeX string as a float.
    Handles: plain numbers, simple \\frac{a}{b}, and a/b fractions.
    Returns None for symbolic expressions that cannot be reduced to a number.
    """
    s = s.strip()
    # Plain integer or decimal (possibly with commas)
    try:
        return float(s.replace(',', ''))
    except ValueError:
        pass
    # \frac{a}{b}  — numerator and denominator are plain numbers
    m = re.fullmatch(r'\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}', s)
    if m:
        try:
            num, den = float(m.group(1)), float(m.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass
    # a/b  — plain ratio
    m = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)', s)
    if m:
        try:
            num, den = float(m.group(1)), float(m.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass
    return None


def _match_element(pred: str, gold: str) -> bool:
    """Compare a single normalized math element numerically then by string."""
    p_val = _to_float(pred)
    g_val = _to_float(gold)
    if p_val is not None and g_val is not None:
        return abs(p_val - g_val) < 1e-6
    return pred == gold


def _match_as_set(pred: str, gold: str) -> bool:
    """
    Order-insensitive comparison for comma-separated answers (e.g. roots of a polynomial).
    Each element is matched numerically where possible, otherwise by string.
    Returns False immediately if element counts differ.
    """
    pred_parts = [_strip_latex(p.strip()) for p in pred.split(',')]
    gold_parts = [_strip_latex(g.strip()) for g in gold.split(',')]
    if len(pred_parts) != len(gold_parts):
        return False
    used = [False] * len(pred_parts)
    for g in gold_parts:
        matched = False
        for i, p in enumerate(pred_parts):
            if not used[i] and _match_element(p, g):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


def score_math(prediction: str, gold: str) -> dict:
    """
    Returns {"exact_match": 0 or 1}.

    Scoring pipeline:
    1. Extract the last \\boxed{} from the prediction (falls back to raw text).
    2. Strip LaTeX formatting from both sides with _strip_latex.
    3. Try numeric comparison first (handles cross-format matches like
       \\frac{1}{2} == 0.5 == 1/2).
    4. For comma-separated answers (e.g. all roots of a polynomial), try
       order-insensitive set matching so that "-1, 7, -3/2" == "-3/2, -1, 7".
    5. Fall back to exact string comparison (handles tuples, complex numbers,
       polynomial expressions that cannot be reduced to floats).

    Gold is already the extracted \\boxed{} content from dataset load time.
    """
    boxed = _extract_math_boxed(prediction)
    pred = _strip_latex(boxed if boxed is not None else prediction)
    gold_norm = _strip_latex(gold)

    # Numeric comparison: catches cross-format matches
    pred_val = _to_float(pred)
    gold_val = _to_float(gold_norm)
    if pred_val is not None and gold_val is not None:
        return {"exact_match": int(abs(pred_val - gold_val) < 1e-6)}

    # Set comparison: catches order-permuted comma-separated answers
    if ',' in gold_norm:
        return {"exact_match": int(_match_as_set(pred, gold_norm))}

    # String comparison: handles symbolic forms that parse to the same string
    return {"exact_match": int(pred == gold_norm)}


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

def _normalize_hotpotqa(text: str) -> str:
    """Official HotpotQA normalization: lowercase, strip articles, remove punctuation."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def score_hotpotqa(prediction: str, gold: str) -> dict:
    """
    Returns {"exact_match": 0 or 1, "f1": float}.
    Uses official HotpotQA normalization before comparison.
    """
    pred_norm = _normalize_hotpotqa(prediction)
    gold_norm = _normalize_hotpotqa(gold)

    em = int(pred_norm == gold_norm)

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        f1 = 0.0
    else:
        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(gold_tokens) if gold_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall)

    return {"exact_match": em, "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Unified scorer
# ---------------------------------------------------------------------------

def score(domain: str, prediction: str, gold: str) -> dict:
    """
    Dispatch to the correct scorer based on domain.
    Always returns a dict with at least "exact_match".
    """
    if domain == "math":
        return score_math(prediction, gold)
    elif domain == "hotpotqa":
        return score_hotpotqa(prediction, gold)
    else:
        raise ValueError(f"Unknown domain: {domain!r}")

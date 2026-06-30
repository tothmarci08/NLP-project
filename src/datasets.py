"""
Dataset loaders for MATH and HotpotQA.

Each loader returns rows with a uniform shape:
  {
    "id": str,
    "question": str,
    "context": dict[str, str],   # paragraph_id -> text (empty for MATH)
    "document_ids": list[str],   # ordered candidate IDs (empty for MATH)
    "gold_answer": str,
  }

Difficulty subsets:
  MATH easy  - competition problems Level 1 and Level 2
  MATH hard  - competition problems Level 4 and Level 5
  HotpotQA easy  - comparison-type questions (single-hop reasoning)
  HotpotQA hard  - bridge-type questions (true multi-hop)
"""

from __future__ import annotations

import random
import re
from typing import Literal

Domain = Literal["math", "hotpotqa"]
Difficulty = Literal["easy", "hard"]


# ---------------------------------------------------------------------------
# \boxed{} extraction helper (used to pull gold answers from MATH solutions)
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str | None:
    """Extract the answer from \\boxed{...} in a MATH solution, handling nested braces."""
    match = re.search(r'\\boxed\{', text)
    if not match:
        return None
    start = match.end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


# ---------------------------------------------------------------------------
# Offline fallback data (used during development / when datasets unavailable)
# ---------------------------------------------------------------------------

_MATH_FALLBACK_EASY = [
    {
        "id": "math_easy_0",
        "question": "Compute $3 + 5 \\cdot 2$.",
        "context": {},
        "document_ids": [],
        "gold_answer": "13",
    },
    {
        "id": "math_easy_1",
        "question": "What is the area of a rectangle with length 6 and width 4?",
        "context": {},
        "document_ids": [],
        "gold_answer": "24",
    },
    {
        "id": "math_easy_2",
        "question": "Simplify: $2^3 + 1$.",
        "context": {},
        "document_ids": [],
        "gold_answer": "9",
    },
    {
        "id": "math_easy_3",
        "question": "If $x = 3$, what is $2x + 4$?",
        "context": {},
        "document_ids": [],
        "gold_answer": "10",
    },
    {
        "id": "math_easy_4",
        "question": "What is $\\frac{12}{4} \\cdot 5$?",
        "context": {},
        "document_ids": [],
        "gold_answer": "15",
    },
]

_MATH_FALLBACK_HARD = [
    {
        "id": "math_hard_0",
        "question": "Find the sum of all positive integer divisors of 12.",
        "context": {},
        "document_ids": [],
        "gold_answer": "28",
    },
    {
        "id": "math_hard_1",
        "question": "If $f(x) = x^2 - 2x + 1$, find $f(3)$.",
        "context": {},
        "document_ids": [],
        "gold_answer": "4",
    },
    {
        "id": "math_hard_2",
        "question": "In how many ways can 4 distinct people be arranged in a line?",
        "context": {},
        "document_ids": [],
        "gold_answer": "24",
    },
    {
        "id": "math_hard_3",
        "question": "What is the value of $\\binom{6}{2}$?",
        "context": {},
        "document_ids": [],
        "gold_answer": "15",
    },
    {
        "id": "math_hard_4",
        "question": "Find the remainder when $17^2$ is divided by 7.",
        "context": {},
        "document_ids": [],
        "gold_answer": "2",
    },
]

_HOTPOTQA_FALLBACK_EASY = [
    {
        "id": "hotpot_easy_0",
        "question": "What is the capital of France?",
        "context": {
            "p0": "France is a country in Western Europe. Its capital city is Paris.",
            "p1": "Germany is located in Central Europe. Berlin is its capital.",
        },
        "document_ids": ["p0", "p1"],
        "gold_answer": "Paris",
    },
    {
        "id": "hotpot_easy_1",
        "question": "Who wrote the novel '1984'?",
        "context": {
            "p0": "George Orwell was an English novelist and essayist. He wrote the dystopian novel 1984.",
            "p1": "Aldous Huxley wrote Brave New World, another famous dystopian novel.",
        },
        "document_ids": ["p0", "p1"],
        "gold_answer": "George Orwell",
    },
    {
        "id": "hotpot_easy_2",
        "question": "What element has the chemical symbol 'O'?",
        "context": {
            "p0": "Oxygen is a chemical element with the symbol O and atomic number 8.",
            "p1": "Gold has the chemical symbol Au and is a precious metal.",
        },
        "document_ids": ["p0", "p1"],
        "gold_answer": "Oxygen",
    },
    {
        "id": "hotpot_easy_3",
        "question": "In which year did World War II end?",
        "context": {
            "p0": "World War II was a global conflict that lasted from 1939 to 1945.",
            "p1": "World War I ended in 1918 with the signing of the Armistice.",
        },
        "document_ids": ["p0", "p1"],
        "gold_answer": "1945",
    },
    {
        "id": "hotpot_easy_4",
        "question": "What is the largest planet in our solar system?",
        "context": {
            "p0": "Jupiter is the largest planet in the solar system, with a mass greater than all other planets combined.",
            "p1": "Saturn is the second largest planet and is known for its prominent ring system.",
        },
        "document_ids": ["p0", "p1"],
        "gold_answer": "Jupiter",
    },
]

_HOTPOTQA_FALLBACK_HARD = [
    {
        "id": "hotpot_hard_0",
        "question": "The director of Pulp Fiction also directed which 1992 heist film that starred Steve Buscemi?",
        "context": {
            "p0": "Pulp Fiction is a 1994 film directed by Quentin Tarantino.",
            "p1": "Reservoir Dogs is a 1992 heist film directed by Quentin Tarantino, starring Steve Buscemi.",
            "p2": "The Usual Suspects is a 1995 crime film directed by Bryan Singer.",
            "p3": "Fargo is a 1996 crime film directed by the Coen Brothers, also starring Steve Buscemi.",
            "p4": "Steve Buscemi is an American actor who has appeared in many crime films.",
            "p5": "Heat is a 1995 crime film directed by Michael Mann.",
        },
        "document_ids": ["p0", "p1", "p2", "p3", "p4", "p5"],
        "gold_answer": "Reservoir Dogs",
    },
    {
        "id": "hotpot_hard_1",
        "question": "Which American president was in office when the Berlin Wall fell, and what party did he belong to?",
        "context": {
            "p0": "The Berlin Wall fell on November 9, 1989.",
            "p1": "George H. W. Bush served as the 41st President of the United States from 1989 to 1993.",
            "p2": "Ronald Reagan was President from 1981 to 1989 and was a member of the Republican Party.",
            "p3": "George H. W. Bush was a member of the Republican Party.",
            "p4": "The Democratic Party is one of the two major political parties in the United States.",
            "p5": "Bill Clinton was the 42nd President, serving from 1993 to 2001.",
        },
        "document_ids": ["p0", "p1", "p2", "p3", "p4", "p5"],
        "gold_answer": "George H. W. Bush, Republican Party",
    },
    {
        "id": "hotpot_hard_2",
        "question": "The author of 'The Great Gatsby' attended which university before dropping out to join the army?",
        "context": {
            "p0": "The Great Gatsby was written by F. Scott Fitzgerald, published in 1925.",
            "p1": "F. Scott Fitzgerald attended Princeton University but left in 1917 to join the U.S. Army.",
            "p2": "Ernest Hemingway, another famous American author, did not attend university.",
            "p3": "Princeton University is a private Ivy League research university in New Jersey.",
            "p4": "Harvard University is located in Cambridge, Massachusetts.",
            "p5": "Yale University is another Ivy League institution in New Haven, Connecticut.",
        },
        "document_ids": ["p0", "p1", "p2", "p3", "p4", "p5"],
        "gold_answer": "Princeton University",
    },
    {
        "id": "hotpot_hard_3",
        "question": "The company that makes the iPhone was founded by Steve Jobs along with which other co-founder who later returned as CEO?",
        "context": {
            "p0": "Apple Inc. is the company that makes the iPhone, founded in 1976.",
            "p1": "Apple was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
            "p2": "Steve Jobs was ousted from Apple in 1985 and returned as CEO in 1997.",
            "p3": "Microsoft was founded by Bill Gates and Paul Allen.",
            "p4": "Tim Cook became CEO of Apple after Steve Jobs resigned in 2011.",
            "p5": "Steve Wozniak left Apple in 1985 but did not return as CEO.",
        },
        "document_ids": ["p0", "p1", "p2", "p3", "p4", "p5"],
        "gold_answer": "Steve Jobs",
    },
    {
        "id": "hotpot_hard_4",
        "question": "In what country was Marie Curie born, and what was the first element she discovered?",
        "context": {
            "p0": "Marie Curie was born on November 7, 1867, in Warsaw, which was then part of the Russian Empire.",
            "p1": "Poland regained independence in 1918. Warsaw is its capital.",
            "p2": "Marie Curie discovered two elements: polonium (named after Poland) and radium.",
            "p3": "Polonium was the first element discovered by Marie Curie, announced in 1898.",
            "p4": "Radium was discovered shortly after polonium in 1898 by the Curies.",
            "p5": "Einstein was born in Germany in 1879.",
        },
        "document_ids": ["p0", "p1", "p2", "p3", "p4", "p5"],
        "gold_answer": "Poland, polonium",
    },
]


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_dataset(
    domain: Domain,
    difficulty: Difficulty,
    n: int = 30,
    seed: int = 42,
    use_fallback: bool = False,
) -> list[dict]:
    """
    Load n rows for (domain, difficulty).

    Args:
        domain: "math" or "hotpotqa"
        difficulty: "easy" or "hard"
        n: number of samples to return
        seed: random seed for reproducibility
        use_fallback: force offline fallback data (for dev/testing)

    Returns:
        List of row dicts with keys: id, question, context, document_ids, gold_answer
    """
    if use_fallback or _is_offline():
        return _load_fallback(domain, difficulty, n, seed)
    return _load_huggingface(domain, difficulty, n, seed)


def _is_offline() -> bool:
    """Check if HuggingFace datasets are available."""
    try:
        import datasets as hf_datasets  # noqa: F401
        return False
    except ImportError:
        return True


def _load_fallback(domain: Domain, difficulty: Difficulty, n: int, seed: int) -> list[dict]:
    mapping = {
        ("math", "easy"): _MATH_FALLBACK_EASY,
        ("math", "hard"): _MATH_FALLBACK_HARD,
        ("hotpotqa", "easy"): _HOTPOTQA_FALLBACK_EASY,
        ("hotpotqa", "hard"): _HOTPOTQA_FALLBACK_HARD,
    }
    rows = mapping[(domain, difficulty)]
    rng = random.Random(seed)
    if len(rows) >= n:
        return rng.sample(rows, n)
    # Repeat if we need more than available (dev mode only)
    repeated = (rows * ((n // len(rows)) + 1))[:n]
    for i, row in enumerate(repeated[len(rows):], start=len(rows)):
        repeated[i] = {**row, "id": f"{row['id']}_dup{i}"}
    return repeated


def _load_huggingface(domain: Domain, difficulty: Difficulty, n: int, seed: int) -> list[dict]:
    import datasets as hf_datasets

    rng = random.Random(seed)

    if domain == "math":
        return _load_math(difficulty, n, rng, hf_datasets)
    else:
        return _load_hotpotqa(difficulty, n, rng, hf_datasets)


def _load_math(difficulty: Difficulty, n: int, rng: random.Random, hf_datasets) -> list[dict]:
    # Level 1-2 = easy, Level 4-5 = hard (Level 3 skipped as medium)
    level_map = {
        "easy": {"Level 1", "Level 2"},
        "hard": {"Level 4", "Level 5"},
    }
    target_levels = level_map[difficulty]

    # qwedsacf/competition_math is a mirror of the original MATH benchmark that
    # merges train+test into a single 12,500-row "train" split.
    ds = hf_datasets.load_dataset("qwedsacf/competition_math", split="train")

    # Shuffle the full dataset BEFORE filtering: the dataset is sorted by type
    # (first several hundred rows are all Algebra), so sequential slicing would
    # yield near-zero subject diversity. Shuffle with the experiment seed first.
    all_items = list(ds)
    rng.shuffle(all_items)

    rows = []
    for item in all_items:
        if item["level"] not in target_levels:
            continue

        # Extract the gold answer from \boxed{} in the reference solution
        gold = _extract_boxed(item["solution"])
        if gold is None:
            continue  # skip problems where extraction fails (very rare)

        row = {
            "id": f"math_{difficulty}_{len(rows)}",
            "question": item["problem"],
            "context": {},
            "document_ids": [],
            "gold_answer": gold,
        }
        rows.append(row)
        if len(rows) >= n:
            break

    return rows[:n]


def _load_hotpotqa(difficulty: Difficulty, n: int, rng: random.Random, hf_datasets) -> list[dict]:
    split = "validation"
    ds = hf_datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    rows = []
    for item in ds:
        # The distractor validation split has level='hard' for every item,
        # so we use the 'type' field as the difficulty discriminator instead:
        #   "comparison" -> easier (compare attributes of two entities, fewer hops)
        #   "bridge"     -> harder (chain facts across multiple documents, true multi-hop)
        question_type = item.get("type", "")

        if difficulty == "easy" and question_type != "comparison":
            continue
        if difficulty == "hard" and question_type != "bridge":
            continue

        # Build context dict from titles + sentences
        context = {}
        doc_ids = []
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
        ):
            pid = re.sub(r'\W+', '_', title)[:40]
            context[pid] = " ".join(sentences)
            doc_ids.append(pid)

        row = {
            "id": item["id"],
            "question": item["question"],
            "context": context,
            "document_ids": doc_ids,
            "gold_answer": item["answer"],
        }
        rows.append(row)
        if len(rows) >= n * 3:
            break

    sampled = rng.sample(rows, min(n, len(rows)))
    return sampled

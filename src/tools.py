"""
Retrieval tool for HotpotQA.

Given a query string and a pool of paragraphs (id -> text),
return the top-k paragraph IDs ranked by TF-IDF cosine similarity.
GSM8K does not use this — it's HotpotQA-only.
"""

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _tf(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    n = len(corpus_tokens)
    df: Counter = Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))
    return {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _tf(tokens)
    return {term: tf[term] * idf.get(term, 0) for term in tf}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(
    query: str,
    documents: dict[str, str],
    top_k: int = 3,
) -> dict[str, str]:
    """
    Retrieve the top_k most relevant paragraphs for a query using TF-IDF cosine similarity.

    Args:
        query: The question or search string.
        documents: Mapping of paragraph_id -> paragraph_text.
        top_k: Number of paragraphs to return.

    Returns:
        Ordered dict of {paragraph_id: text} for the top_k results.
    """
    if not documents:
        return {}

    doc_ids = list(documents.keys())
    corpus = [_tokenize(documents[pid]) for pid in doc_ids]
    query_tokens = _tokenize(query)

    idf = _idf(corpus + [query_tokens])

    query_vec = _tfidf_vector(query_tokens, idf)
    scored = []
    for i, pid in enumerate(doc_ids):
        doc_vec = _tfidf_vector(corpus[i], idf)
        sim = _cosine(query_vec, doc_vec)
        scored.append((sim, pid))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_ids = [pid for _, pid in scored[:top_k]]

    return {pid: documents[pid] for pid in top_ids}

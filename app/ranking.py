import re

import numpy as np


def clean_query(query: str, strip_words: list[str]) -> str:
    """Remove filler like "a song about" that never appears in lyrics.

    "a sad song about losing someone"  ->  "sad losing someone"
    Falls back to the original query if stripping would leave nothing.
    """
    q = query.strip()
    for phrase in sorted(strip_words or [], key=len, reverse=True):
        q = re.sub(rf"\b{re.escape(phrase)}\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip(" ,.-")
    return q if q else query.strip()


def best_chunk(query_vec: np.ndarray, chunk_vecs: np.ndarray,
               top_n: int = 1) -> tuple[float, int]:
    """Chunk-level score for one song, and the position of its best chunk.

    top_n=1 -> cosine of the single best chunk.
    top_n>1 -> average of the n best chunks (less fooled by one short,
               repetitive line like "love love love").
    """
    sims = chunk_vecs @ query_vec
    j = int(np.argmax(sims))
    if top_n <= 1 or len(sims) == 1:
        return float(sims[j]), j
    return float(np.sort(sims)[::-1][:top_n].mean()), j


def expand_query(query_vec: np.ndarray, top_song_vecs: np.ndarray, beta: float) -> np.ndarray:
    """Pseudo-relevance feedback (Rocchio): nudge the query toward the
    centre of its own top results, then search again. Helps when the query
    is one short word ("loneliness") by borrowing vocabulary from songs that
    clearly match it."""
    centre = top_song_vecs.mean(axis=0)
    centre /= np.linalg.norm(centre) + 1e-12
    q = query_vec + beta * centre
    return (q / (np.linalg.norm(q) + 1e-12)).astype(np.float32)


def combine(chunk_score, mean_score, title_score, weights: dict):
    """Weighted blend of the three similarity signals (all cosine, -1..1).

    Weights are normalised to sum to 1, so the result stays on the same
    cosine scale and is directly comparable to model_v1 scores.
    """
    w_c = float(weights.get("chunk", 0.0))
    w_m = float(weights.get("mean", 0.0))
    w_t = float(weights.get("title", 0.0))
    total = w_c + w_m + w_t
    if total <= 0:
        raise ValueError("search.weights must contain at least one positive weight")
    return (w_c * np.asarray(chunk_score) + w_m * np.asarray(mean_score)
            + w_t * np.asarray(title_score)) / total


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def to_probability(scores) -> np.ndarray:
    """Cross-encoders return either raw logits or (in newer
    sentence-transformers versions) values already passed through a sigmoid.
    Normalise both to a 0-1 relevance probability."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size and s.min() >= 0.0 and s.max() <= 1.0:
        return s
    return sigmoid(s)


def match_percent(cosine, low: float = 0.30, high: float = 0.75):
    """Map a raw cosine onto an easier-to-read 0-100 "match %".

    This is ONLY a display rescale (it changes no ranking): for MiniLM, a
    short query vs. lyrics almost never exceeds ~0.8 cosine even for a
    perfect match, so `high` is treated as 100%. Always report the raw
    cosine alongside it.
    """
    c = np.asarray(cosine, dtype=np.float64)
    return np.clip((c - low) / (high - low), 0.0, 1.0) * 100.0

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.search_engine import SemanticSearchEngine  # noqa: E402

EVAL_QUERIES_PATH = Path("configs/eval_queries.yaml")  # override with --eval-file
CANDIDATES_OUT_PATH = Path("out/eval_candidates.csv")
TOP_K = 10


def load_eval_queries():
    if not EVAL_QUERIES_PATH.exists():
        # Minimal starter set of queries -- NOT labels. You still have to
        # fill in `relevant_songs` yourself after reading real results.
        starter = {
            "queries": [
                {"query": "sad heartbreak", "relevant_songs": []},
                {"query": "romantic love song", "relevant_songs": []},
                {"query": "falling in love", "relevant_songs": []},
                {"query": "loneliness", "relevant_songs": []},
                {"query": "missing someone", "relevant_songs": []},
                {"query": "friendship", "relevant_songs": []},
                {"query": "feeling confident and powerful", "relevant_songs": []},
                {"query": "party celebration", "relevant_songs": []},
                {"query": "losing someone you love", "relevant_songs": []},
            ]
        }
        EVAL_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_QUERIES_PATH, "w") as f:
            yaml.safe_dump(starter, f, sort_keys=False)
        print(f"Created starter query list at {EVAL_QUERIES_PATH} "
              f"(relevant_songs left empty -- see this file's usage docstring).")
        return starter["queries"]

    with open(EVAL_QUERIES_PATH) as f:
        return yaml.safe_load(f)["queries"]


def dump_candidates(engine, queries):
    rows = []
    for q in queries:
        results = engine.search(q["query"], top_k=TOP_K)
        for r in results:
            rows.append({
                "query": q["query"],
                "rank": r["rank"],
                "artist": r["artist"],
                "song": r["song"],
                "similarity_score": r["similarity_score"],
                "lyrics_excerpt": r["lyrics_excerpt"],
                "relevant": "",  # <- fill with 1 or 0 by hand
            })
    df = pd.DataFrame(rows)
    CANDIDATES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CANDIDATES_OUT_PATH, index=False)
    print(f"Wrote {len(df)} candidate rows to {CANDIDATES_OUT_PATH}.")
    print("Open it, mark the `relevant` column (1/0) for each row, then "
          "copy the artist/song pairs you marked 1 into "
          "configs/eval_queries.yaml under the matching query's "
          "`relevant_songs` list.")


def precision_at_k(retrieved, relevant_set, k):
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if (r["artist"], r["song"]) in relevant_set)
    return hits / len(top)


def recall_at_k(retrieved, relevant_set, k):
    if not relevant_set:
        return None  # undefined -- no labels for this query
    top = retrieved[:k]
    hits = sum(1 for r in top if (r["artist"], r["song"]) in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved, relevant_set):
    for r in retrieved:
        if (r["artist"], r["song"]) in relevant_set:
            return 1.0 / r["rank"]
    return 0.0


def run_metrics(engine, queries, label="semantic search"):
    print(f"\n=== Evaluating: {label} ===")
    labeled_queries = [q for q in queries if q.get("relevant_songs")]
    if not labeled_queries:
        print("No queries have labeled `relevant_songs` yet -- nothing to "
              "score. Run with --dump-only first, label the results, then "
              "re-run.")
        return

    p_scores, r_scores, rr_scores, nd_scores = [], [], [], []
    for q in labeled_queries:
        relevant_set = {tuple(pair) for pair in q["relevant_songs"]}
        retrieved = engine.search(q["query"], top_k=TOP_K)

        p = precision_at_k(retrieved, relevant_set, TOP_K)
        r = recall_at_k(retrieved, relevant_set, TOP_K)
        rr = reciprocal_rank(retrieved, relevant_set)
        nd = ndcg_at_k(retrieved, relevant_set, TOP_K)
        nd_scores.append(nd)

        p_scores.append(p)
        if r is not None:
            r_scores.append(r)
        rr_scores.append(rr)

        print(f"  '{q['query']}'  ->  P@{TOP_K}={p:.2f}  "
              f"R@{TOP_K}={'n/a' if r is None else f'{r:.2f}'}  RR={rr:.2f}  nDCG={nd:.2f}")

    print(f"\nMean P@{TOP_K}: {sum(p_scores)/len(p_scores):.3f}")
    if r_scores:
        print(f"Mean R@{TOP_K}: {sum(r_scores)/len(r_scores):.3f}")
    print(f"MRR:        {sum(rr_scores)/len(rr_scores):.3f}")
    print(f"nDCG@{TOP_K}:   {sum(nd_scores)/len(nd_scores):.3f}")
    print(f"(computed over {len(labeled_queries)} labeled queries)")


def ndcg_at_k(retrieved, relevant_set, k):
    import math
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(retrieved[:k])
              if (r["artist"], r["song"]) in relevant_set)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_set))))
    return dcg / ideal if ideal else 0.0


def similarity_stats(engine, queries):
    """Average scores of the #1 result and of the top-10 (labels not needed)."""
    stats = {"similarity_score": ([], []), "match_percent": ([], []), "relevance": ([], [])}
    for q in queries:
        res = engine.search(q["query"], top_k=TOP_K)
        if not res:
            continue
        for key, (top1, top10) in stats.items():
            vals = [r.get(key) for r in res if r.get(key) is not None]
            if vals:
                top1.append(vals[0])
                top10.append(sum(vals) / len(vals))
    labels = {"similarity_score": "raw cosine", "match_percent": "match %",
              "relevance": "re-ranker relevance"}
    for key, (top1, top10) in stats.items():
        if top1:
            print(f"{labels[key]:>20}:  #1 = {sum(top1)/len(top1):.3f}   "
                  f"top-{TOP_K} avg = {sum(top10)/len(top10):.3f}")


V1_SETTINGS = {"weights": {"chunk": 0.0, "mean": 1.0, "title": 0.0}, "strip_query_words": [],
               "query_expansion": {"enabled": False}, "minimum_score": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-only", action="store_true",
                         help="only write candidate results for manual labeling")
    parser.add_argument("--baseline", action="store_true",
                         help="also evaluate a TF-IDF keyword-search baseline")
    parser.add_argument("--compare", action="store_true",
                         help="also score the same index with model_v1 settings "
                              "(mean vector only) for a before/after table")
    parser.add_argument("--weights", default=None,
                         help="override search weights, e.g. --weights 0.6,0.25,0.15 "
                              "(chunk,mean,title) -- handy for tuning")
    parser.add_argument("--no-rerank", action="store_true",
                         help="evaluate model_v2 with the cross-encoder re-ranker switched off "
                              "(run with and without to see if the re-ranker helps)")
    parser.add_argument("--eval-file", default=None,
                         help="labels file (default configs/eval_queries.yaml; try "
                              "configs/eval_queries_pooled.yaml for the fairer pooled labels)")
    args = parser.parse_args()

    global EVAL_QUERIES_PATH
    if args.eval_file:
        EVAL_QUERIES_PATH = Path(args.eval_file)
    queries = load_eval_queries()
    engine = SemanticSearchEngine()
    if args.no_rerank:
        engine.reranker = None
    if args.weights:
        c, m, t = (float(x) for x in args.weights.split(","))
        engine.search_cfg["weights"] = {"chunk": c, "mean": m, "title": t}

    if args.dump_only:
        dump_candidates(engine, queries)
        return

    if args.compare:
        saved = {k: engine.search_cfg.get(k) for k in V1_SETTINGS}
        saved_reranker = engine.reranker
        engine.search_cfg.update(V1_SETTINGS)
        engine.reranker = None
        run_metrics(engine, queries, label="model_v1 settings (mean vector only)")
        similarity_stats(engine, queries)
        engine.search_cfg.update(saved)
        engine.reranker = saved_reranker

    w = engine.search_cfg["weights"]
    rr_state = "on" if engine.reranker is not None else "off"
    run_metrics(engine, queries, label=f"model_v2 (chunk={w['chunk']}, mean={w['mean']}, "
                                       f"title={w['title']}, re-ranker {rr_state})")
    similarity_stats(engine, queries)

    if args.baseline:
        run_tfidf_baseline(queries)


def run_tfidf_baseline(queries):
    """Optional: a plain TF-IDF keyword-overlap baseline, for comparison
    against the semantic engine. This does NOT use the project's saved
    embeddings/index -- it's a separate, independent baseline computed
    directly from the processed lyrics.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    df = pd.read_csv(config["data"]["processed_path"])

    vectorizer = TfidfVectorizer(stop_words="english", max_features=60_000)
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    class TfidfBaselineEngine:
        def search(self, query, top_k=10):
            qvec = vectorizer.transform([query])
            scores = (tfidf_matrix @ qvec.T).toarray().ravel()
            top_idx = scores.argsort()[::-1][:top_k]
            return [
                {
                    "rank": rank + 1,
                    "artist": df.iloc[i]["artist"],
                    "song": df.iloc[i]["song"],
                    "similarity_score": float(scores[i]),
                }
                for rank, i in enumerate(top_idx)
            ]

    queries_reloaded = load_eval_queries()
    run_metrics(TfidfBaselineEngine(), queries_reloaded, label="TF-IDF keyword baseline")


if __name__ == "__main__":
    main()

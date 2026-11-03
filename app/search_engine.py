"""model_v2 search engine.

Pipeline for one query:
  1. clean the query ("a sad song about X" -> "sad X") and embed it
  2. gather candidate songs from three FAISS indexes
       - chunks.index : songs owning one of the best-matching lyric chunks
       - songs.index  : songs whose whole-song (mean) vector matches
       - titles.index : songs whose title matches
  3. score every candidate exactly:
       final = w_chunk * best chunk cosine + w_mean * mean cosine + w_title * title cosine
  4. query expansion: move the query toward its top-5 results and repeat 2-3
  5. (optional) re-rank the top candidates with a cross-encoder
"""
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import yaml

APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from text_chunking import lyrics_to_chunks  # noqa: E402

from app.ranking import (best_chunk, clean_query, combine, expand_query,  # noqa: E402
                         match_percent, to_probability)  # noqa: E402


class SearchEngineError(Exception):
    """Raised when the engine can't load its required files, or a query
    can't be answered. app/main.py turns this into a clean HTTP error."""


class SemanticSearchEngine:
    def __init__(self, config_path: str = "configs/config.yaml"):
        config_path = PROJECT_ROOT / config_path
        if not config_path.exists():
            raise SearchEngineError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        emb = self.config["embedding"]
        self.search_cfg = self.config["search"]
        self.lyrics_cfg = self.config["lyrics"]
        self.dimension = emb["dimension"]
        self.model_name = emb["model_name"]
        self.query_prefix = emb.get("query_prefix", "") or ""

        out = self.config["output"]
        paths = {k: PROJECT_ROOT / out[k] for k in
                 ["chunk_index_path", "song_index_path", "title_index_path",
                  "chunk_offsets_path", "metadata_path"]}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise SearchEngineError(
                "Search index not found. Run, in order:\n"
                "  python scripts/prepare_dataset.py\n"
                "  python scripts/generate_embeddings.py\n"
                "  python scripts/build_index.py\n"
                f"(missing: {', '.join(missing)})"
            )

        print("Loading FAISS indexes ...")
        self.chunk_index = faiss.read_index(str(paths["chunk_index_path"]))
        song_index = faiss.read_index(str(paths["song_index_path"]))
        title_index = faiss.read_index(str(paths["title_index_path"]))
        self.song_index, self.title_index = song_index, title_index
        # Keep the small per-song matrices in RAM for exact re-scoring.
        self.song_vecs = song_index.reconstruct_n(0, song_index.ntotal)
        self.title_vecs = title_index.reconstruct_n(0, title_index.ntotal)
        self.offsets = np.load(paths["chunk_offsets_path"])
        self.chunk_owner = np.repeat(np.arange(len(self.offsets) - 1),
                                     np.diff(self.offsets))

        print("Loading metadata ...")
        self.metadata = pd.read_csv(paths["metadata_path"])

        n = len(self.metadata)
        if not (song_index.ntotal == title_index.ntotal == n
                and self.offsets[-1] == self.chunk_index.ntotal):
            raise SearchEngineError(
                "Indexes and metadata are out of sync -- re-run build_index.py."
            )
        # Kept for app/main.py, which shows the song count.
        self.index = song_index

        print(f"Loading embedding model: {self.model_name} ...")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise SearchEngineError(
                "sentence-transformers is not installed. "
                "Run: pip install -r requirements.txt"
            ) from e
        model_path = PROJECT_ROOT / self.model_name
        self.model = SentenceTransformer(str(model_path) if model_path.exists()
                                         else self.model_name)

        self.reranker = None
        rr = self.config.get("rerank", {}) or {}
        if rr.get("enabled"):
            from sentence_transformers import CrossEncoder
            print(f"Loading re-ranker: {rr['model_name']} "
                  f"(downloads ~90MB the first time) ...")
            try:
                self.reranker = CrossEncoder(rr["model_name"])
            except Exception as e:  # no internet, bad name, ...
                print(f"WARNING: re-ranker could not load ({e}); continuing without it.")
                self.reranker = None

        print(f"Search engine ready: {n} songs, {self.chunk_index.ntotal} lyric chunks, "
              f"{self.dimension}-dim embeddings.")

    # ------------------------------------------------------------------ #
    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([self.query_prefix + query],
                                convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    def _chunk_vectors(self, song_id: int) -> np.ndarray:
        start, end = int(self.offsets[song_id]), int(self.offsets[song_id + 1])
        return self.chunk_index.reconstruct_n(start, end - start)

    def _candidates(self, qv: np.ndarray) -> np.ndarray:
        cfg = self.search_cfg
        _, chunk_ids = self.chunk_index.search(
            qv, min(cfg.get("chunk_candidates", 400), self.chunk_index.ntotal))
        k = min(cfg.get("song_candidates", 100), self.song_index.ntotal)
        _, song_ids = self.song_index.search(qv, k)
        _, title_ids = self.title_index.search(qv, k)
        ids = np.concatenate([self.chunk_owner[chunk_ids[0][chunk_ids[0] >= 0]],
                              song_ids[0], title_ids[0]])
        return np.unique(ids[ids >= 0])

    def _matched_text(self, song_id: int, chunk_pos: int) -> str:
        text = self.metadata.iloc[song_id].get("text", "")
        if not isinstance(text, str):
            return ""
        chunks = lyrics_to_chunks(text, self.lyrics_cfg["sentences_per_chunk"],
                                  self.lyrics_cfg["sentence_overlap"])
        return chunks[chunk_pos] if chunk_pos < len(chunks) else ""

    def _score(self, q: np.ndarray, cands: np.ndarray) -> dict:
        """Exact chunk / whole-song / title cosines for every candidate."""
        top_n = int(self.search_cfg.get("chunk_top_n", 1))
        chunk_s = np.empty(len(cands), np.float32)
        chunk_pos = np.empty(len(cands), np.int64)
        for i, sid in enumerate(cands):
            chunk_s[i], chunk_pos[i] = best_chunk(q, self._chunk_vectors(int(sid)), top_n)
        mean_s = self.song_vecs[cands] @ q
        title_s = self.title_vecs[cands] @ q
        final = combine(chunk_s, mean_s, title_s, self.search_cfg["weights"])
        return {"cands": cands, "final": final, "chunk": chunk_s, "pos": chunk_pos,
                "mean": mean_s, "title": title_s}

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not query or not query.strip():
            raise SearchEngineError("Query text must not be empty.")
        if top_k < 1:
            raise SearchEngineError("top_k must be at least 1.")

        cleaned = clean_query(query, self.search_cfg.get("strip_query_words", []))
        qv = self.embed_query(cleaned)
        q = qv[0]
        s = self._score(q, self._candidates(qv))

        # Optional query expansion (pseudo-relevance feedback): rank with an
        # expanded query, but REPORT similarity to what the user typed.
        prf = self.search_cfg.get("query_expansion", {}) or {}
        if prf.get("enabled"):
            m = int(prf.get("feedback_songs", 5))
            top_ids = s["cands"][np.argsort(-s["final"])[:m]]
            q2 = expand_query(q, self.song_vecs[top_ids], float(prf.get("beta", 0.5)))
            rank_score = self._score(q2, self._candidates(q2[None, :]))
            s = self._score(q, rank_score["cands"])   # display scores vs. original query
            s["rank"] = rank_score["final"]
        else:
            s["rank"] = s["final"]

        cands, final = s["cands"], s["final"]
        rank = s["rank"].astype(np.float64)
        order = np.argsort(-rank)
        rr = self.config.get("rerank", {}) or {}
        relevance = np.full(len(cands), np.nan)
        if self.reranker is not None:
            # re-rank at least as many songs as the user asked for
            top = order[: max(int(rr.get("candidates", 50)), top_k)]
            pairs = [(cleaned, self._matched_text(int(cands[i]), int(s["pos"][i])))
                     for i in top]
            relevance[top] = to_probability(self.reranker.predict(pairs))
            w = float(rr.get("weight", 0.5))
            rank[top] = (1 - w) * rank[top] + w * relevance[top]
            order = np.concatenate([top[np.argsort(-rank[top])], order[len(top):]])

        disp = self.config.get("display", {}) or {}
        pct = match_percent(final, float(disp.get("match_low", 0.30)),
                            float(disp.get("match_high", 0.75)))
        min_score = float(self.search_cfg.get("minimum_score", 0.0))
        min_rel = float(rr.get("minimum_relevance", 0.0))
        results = []
        for i in order:
            if len(results) >= top_k:
                break
            if final[i] < min_score:
                continue
            if not np.isnan(relevance[i]) and relevance[i] < min_rel:
                continue
            sid = int(cands[i])
            row = self.metadata.iloc[sid]
            results.append({
                "rank": len(results) + 1,
                "artist": row["artist"],
                "song": row["song"],
                "link": row["link"],
                # raw cosine (blend of the three signals) -- the honest number
                "similarity_score": round(float(final[i]), 4),
                # cosine rescaled for display (0.30 -> 0%, 0.75 -> 100%)
                "match_percent": int(round(float(pct[i]))),
                # cross-encoder relevance probability 0-1 (None if re-ranker is off)
                "relevance": None if np.isnan(relevance[i]) else round(float(relevance[i]), 4),
                "scores": {
                    "best_chunk": round(float(s["chunk"][i]), 4),
                    "whole_song": round(float(s["mean"][i]), 4),
                    "title": round(float(s["title"][i]), 4),
                },
                "matched_lyrics": re.sub(r"\s+", " ",
                                         self._matched_text(sid, int(s["pos"][i]))),
                "lyrics_excerpt": row["lyrics_excerpt"],
            })
        return results

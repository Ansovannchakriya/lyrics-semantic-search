# Lyrics Semantic Search: Finding Songs by Meaning

Describe a mood or theme in plain English, such as *"a sad song about losing someone
you love"*, and get back the songs whose **lyrics** mean the closest thing, even
when they share none of your words.

Built with sentence-transformers, FAISS and FastAPI over the Spotify Million
Song Dataset (57,648 songs).

**Author:** Sovannchakriya AN (GitHub: [Ansovannchakriya](https://github.com/Ansovannchakriya))

## Quick Start
```bash
git clone https://github.com/Ansovannchakriya/lyrics-semantic-search.git
cd lyrics-semantic-search
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows  (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt

# 1. download spotify_millsongdata.csv from Kaggle into data/raw/   (see section 21)
python scripts/prepare_dataset.py
python scripts/generate_embeddings.py   # slow step, 30-60 min on CPU
python scripts/build_index.py
python -m uvicorn app.main:app --reload # open http://127.0.0.1:8000
```
The dataset, embeddings, indexes and models are **not stored in this repo**
(they are hundreds of MB; see `.gitignore`). The commands above rebuild them.

## 1. Project Title
Lyrics Semantic Search: Finding Songs by Meaning, a lyrics-based "vibe search" engine.

## 2. Project Overview
This is a search engine for songs that works by *meaning*, not by exact
keyword matching. You describe a mood, emotion, or theme in plain English,
and the engine returns songs whose **lyrics** are semantically closest to
that description.

## 3. Problem Statement
Traditional song search only works if you already know the artist or exact
title. There's no easy way to search by "the vibe" of a song — e.g. "a sad
song about losing someone I love" — using keyword search alone, because the
words in your query often don't literally appear in the lyrics of a
matching song.

## 4. Objective
Build a system that:
- takes a natural-language description of a song's mood/theme,
- embeds it into the same vector space as every song's lyrics,
- returns the top-K most semantically similar songs, ranked by similarity.

## 5. Dataset Description
**Spotify Million Song Dataset** (`spotify_millsongdata.csv`, from Kaggle),
~57,650 songs with lyrics. This project is not affiliated with Spotify; it only
uses this public dataset.

## 6. Exact Dataset Columns
| Column | Meaning |
|---|---|
| `artist` | Artist name |
| `song` | Track title |
| `link` | Relative path to the original lyrics page (lyricsfreak.com) |
| `text` | Full lyrics |

No other columns exist in the raw data. In particular, there is **no
genre, tempo, BPM, loudness, danceability, energy, or acousticness
column** — see Limitations below.

## 7. Dataset Limitations
- **No audio features at all.** This system only ever sees text. It
  cannot know how a song actually *sounds* — its tempo, energy,
  instrumentation, or production.
- **No genre labels.** A query like "a pop song about love" is answered
  purely from lyrical content, not genre. The system is not doing genre
  classification and does not claim to.
- **`link` is a relative path**, not a full URL — the app prefixes it with
  `https://www.lyricsfreak.com` to make it clickable.
- Two exact duplicate `(artist, song)` rows exist in the raw file and are
  dropped during preprocessing.

## 8. System Architecture

```
data/raw/spotify_millsongdata.csv
              │
              ▼
   scripts/prepare_dataset.py   (clean, dedupe)
              │
              ▼
     data/processed/songs.csv
              │
              ▼
scripts/finetune_model.py       (OPTIONAL: fine-tune the model on the lyrics)
              │
              ▼
scripts/generate_embeddings.py  (chunk vectors + mean song vectors + title vectors)
              │
              ▼
data/processed/chunk_embeddings.npy, embeddings.npy, title_embeddings.npy, metadata.csv
              │
              ▼
   scripts/build_index.py       (3 FAISS indexes)
              │
              ▼
     out/model_v2/chunks.index, songs.index, titles.index, metadata.csv
              │
              ▼
    app/search_engine.py        (loads model + index ONCE)
              │
              ▼
        app/main.py             (FastAPI: GET / , POST /search)
              │
              ▼
         Web browser
```

## 9. Folder Structure
```
lyrics-semantic-search/
├── configs/
│   ├── config.yaml            # all tunable settings, read by every script
│   ├── eval_queries.yaml      # labels made from model_v1's top-10
│   └── eval_queries_pooled.yaml  # fairer pooled labels (see Evaluation)
├── scripts/
│   ├── prepare_dataset.py
│   ├── text_chunking.py       # sentence splitting + chunking helper
│   ├── finetune_model.py      # optional self-supervised fine-tuning
│   ├── generate_embeddings.py
│   ├── build_index.py
│   └── evaluate_search.py
├── data/
│   ├── raw/spotify_millsongdata.csv
│   └── processed/songs.csv, metadata.csv, embeddings.npy
├── out/
│   ├── finetuned_model/       # created by finetune_model.py
│   └── model_v2/chunks.index, songs.index, titles.index, metadata.csv, config_used.yaml
├── app/
│   ├── main.py
│   ├── search_engine.py
│   ├── ranking.py             # scoring helpers (blend, best chunk, query expansion)
│   ├── templates/index.html
│   └── static/style.css, script.js
├── requirements.txt
├── README.md
└── .gitignore             # keeps data, models, indexes and .venv out of git
```

## 10. Technologies Used
Python, FastAPI, Uvicorn, Jinja2, pandas, numpy, scikit-learn (baseline
only), sentence-transformers, FAISS, PyYAML, vanilla HTML/CSS/JS.

## 11. Sentence Transformer
`sentence-transformers/all-MiniLM-L6-v2` (optionally fine-tuned on the lyrics, see section 24), a bi-encoder that maps text to **384-dimensional**
vectors, trained so that semantically similar sentences land close
together in vector space.

## 12. Why `all-MiniLM-L6-v2`
- Trained specifically for semantic similarity — captures meaning, not
  just word overlap, which is exactly what "vibe" queries need.
- Small and fast enough to embed ~57k songs on CPU in a reasonable amount
  of time; no GPU required.
- 384 dimensions keeps the FAISS index small (~85MB for the full dataset)
  and search latency negligible.
- The standard, well-documented general-purpose sentence-embedding model —
  a defensible, easily justified choice for a project at this scale.

## 13. Sentence Splitting
Lyrics are messy: many lines carry no terminal punctuation at all. A naive
`text.split('.')` performs badly on this kind of text. `scripts/text_chunking.py`
splits primarily on line breaks (lyrics are already naturally broken into
short phrase units), and only falls back to punctuation-based splitting
(`. ! ?`) within unusually long single lines that likely contain more than
one sentence. This degrades gracefully whether or not the source lyrics
are properly punctuated.

## 14. Sentence-Based Chunking
Sentences are grouped into overlapping chunks (default: 5 sentences per
chunk, 1 sentence of overlap, both configurable in `config.yaml`), e.g. for
9 sentences: chunk 1 = sentences 1–5, chunk 2 = sentences 5–9. Chunking
(rather than truncating to the first N tokens) means a song's emotional
peak in a later verse or chorus still contributes to its embedding, instead
of being cut off.

## 15. Scoring a Song (model_v2)
model_v1 averaged every chunk into one vector per song. That is robust, but a
song whose chorus matches the query perfectly gets diluted by neutral verses.
model_v2 keeps **three** signals per song and blends them:

| Signal | What it is | Index |
|---|---|---|
| `whole_song` | mean of all chunk vectors (the v1 vector) | `songs.index` |
| `best_chunk` | cosine of the single best-matching lyric chunk | `chunks.index` |
| `title` | cosine of the song title | `titles.index` |

```
score = 0.65 * whole_song + 0.20 * best_chunk + 0.15 * title     (configs/config.yaml -> search.weights)
```
The weights were tuned on the evaluation queries. A heavy chunk weight raises
the similarity numbers but lets short repetitive lines ("love love love")
win, so the whole-song vector stays dominant.

**Query expansion (pseudo-relevance feedback).** After a first search, the
query vector is nudged toward the centre of its top-5 results and the search
runs again (`search.query_expansion`). Ranking uses the expanded query, but the
`similarity_score` shown is always similarity to *what you typed*.

**Matched lines.** Every result returns `matched_lyrics` (the best chunk) and a
`scores` breakdown, which the web page shows so you can see *why* a song matched.

**Re-ranker (on by default).** The top 50 songs are re-read by a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`), which sees the query and the matched
lyrics *together* and outputs a 0-1 **relevance** probability. Ranking blends
it 50/50 with the cosine score. It downloads ~90MB the first time; if it can't
load, the app prints a warning and continues without it. Set `rerank.enabled:
false` if searches feel slow. Check that it helps on your data with:
`python scripts/evaluate_search.py --eval-file configs/eval_queries_pooled.yaml`
vs. the same command with `--no-rerank`.

**Reading the scores.** Each result returns three numbers:

| Field | Meaning |
|---|---|
| `similarity_score` | raw cosine (blend of chunk / whole-song / title). For MiniLM, a short query vs. lyrics rarely exceeds ~0.8 even for a perfect match; **above 0.6 is a strong match**. |
| `relevance` | cross-encoder probability 0-1 (strong matches usually 0.8+). Shown as the main score on the page. |
| `match_percent` | the cosine rescaled for display only (0.30 -> 0%, 0.75 -> 100%, `display:` in config.yaml). Does not change ranking. |

Results with cosine below `search.minimum_score` (0.45) are hidden as weak matches.

## 16. Embedding Normalization
After mean-pooling, each song vector is L2-normalized (vector divided by
its own length, to unit length 1.0). Query vectors are normalized the same
way at search time. This is what makes an inner-product FAISS index
mathematically equivalent to cosine similarity.

## 17. FAISS
Three exact (brute-force) indexes. `chunks.index` holds every lyric chunk
(~10 per song) as float16 (`IndexScalarQuantizer QT_fp16`), which halves its
memory with no practical loss; `songs.index` and `titles.index` are
`IndexFlatIP`. At this scale (~57k songs / ~600k chunks), exact search is
still fast, so approximate indexes (HNSW/IVF) aren't needed.

## 18. Cosine Similarity
Because every stored and query vector is L2-normalized, `IndexFlatIP`'s
inner product score *is* the cosine similarity — no separate normalization
step is needed at query time beyond normalizing the query vector itself.
Cosine similarity is the right metric here because it measures the angle
(semantic direction) between vectors regardless of their magnitude, which
can otherwise vary with lyric length or word choice.

## 19. Installation

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Use `python -m uvicorn` / `python -m pip` rather than `uvicorn` / `pip` directly:
if the project folder (and its `.venv`) is ever moved, the `.exe` launchers break
with *"Fatal error in launcher"*, but `python -m ...` keeps working.

If PowerShell blocks the activation script (execution policy error), either:
- run `powershell -ExecutionPolicy Bypass -File .venv\Scripts\Activate.ps1`, or
- skip activation and call the venv's Python directly:
  `.venv\Scripts\python.exe -m pip install -r requirements.txt` and later
  `.venv\Scripts\python.exe -m uvicorn app.main:app --reload`.

## 20. Configuration
All tunable values live in `configs/config.yaml` (paths, model name,
embedding dimension, chunking parameters, FAISS settings, server host/port).
Every script reads this file rather than hardcoding values — change the
chunk size or model name in one place and every script picks it up.

## 21. Dataset Setup
Download `spotify_millsongdata.csv` from Kaggle (search "Spotify Million Song
Dataset") and place it at:
```
data/raw/spotify_millsongdata.csv
```
**What is (and isn't) in the GitHub repo.** Code, configs, evaluation labels and
docs are committed. These are generated locally and ignored by `.gitignore`:

| Path | What | Rebuilt by |
|---|---|---|
| `data/raw/*.csv` | Kaggle dataset (~75MB) | download from Kaggle |
| `data/processed/*` | cleaned CSV + embeddings (~600MB) | `prepare_dataset.py`, `generate_embeddings.py` |
| `out/model_*/` | FAISS indexes + metadata | `build_index.py` |
| `out/finetuned_model/` | fine-tuned model weights | `finetune_model.py` |
| `.venv/` | Python environment | `python -m venv .venv` + `pip install` |

Empty `data/raw/`, `data/processed/` and `out/` folders are kept with `.gitkeep` files.

## 22. Preprocessing
```bash
python scripts/prepare_dataset.py
```
Loads the raw CSV, verifies the required columns exist, drops rows with
missing artist/song/lyrics, cleans whitespace (keeping line breaks, which
the chunker relies on), removes duplicate `(artist, song)` rows, and saves
`data/processed/songs.csv`. Prints before/after row counts and lyric-length
statistics.

**Actual output on the full dataset:**
```
Original rows:              57650
Removed (missing fields):   0
Removed (empty lyrics):     0
Removed (duplicate songs):  2
Rows after cleaning:        57648
Average lyric length (chars): 1088.3
Maximum lyric length (chars): 3837
```

## 23. Retraining (full pipeline)
```bash
python scripts/prepare_dataset.py          # only if songs.csv is missing
python scripts/finetune_model.py           # OPTIONAL, see section 24
python scripts/generate_embeddings.py      # slow step: ~600k chunks
python scripts/build_index.py
python scripts/evaluate_search.py --compare --eval-file configs/eval_queries_pooled.yaml
python -m uvicorn app.main:app --reload
```
Quick smoke test on 2,000 songs first: `python scripts/generate_embeddings.py --limit 2000`
then `build_index.py` (then run the full thing).

`generate_embeddings.py` now saves chunk vectors (`chunk_embeddings.npy`,
float16), `chunk_offsets.npy` (which chunks belong to which song), the mean
song vectors (`embeddings.npy`, same as v1) and `title_embeddings.npy`.
`build_index.py` checks that all of them line up before writing `out/model_v2/`.

## 24. Fine-tuning the Model Yourself
`scripts/finetune_model.py` adapts the sentence-transformer to song lyrics
**without hand labels** (self-supervised contrastive learning):

- training pairs: *(song title → a chunk of that song)* and
  *(hook line, i.e. the most repeated line → another chunk of the same song)*
- loss: Multiple-Negatives Ranking Loss. In each batch of 32 pairs the model
  must pick the right chunk for every anchor out of all 32 chunks, so it learns
  which short phrase describes which lyric passage.
- 2% of songs are held out and it prints **validation top-1 accuracy before
  and after** training, so you can see whether it helped.
- songs in `eval_queries*.yaml` are excluded from training so evaluation stays fair.

```bash
python scripts/finetune_model.py --max-pairs 5000     # quick test (~5-10 min CPU)
python scripts/finetune_model.py                      # full: 60k pairs, 1 epoch
```
Then set `embedding.model_name: "out/finetuned_model"` in `config.yaml` and
re-run `generate_embeddings.py` + `build_index.py`. All knobs are under `finetune:` in
`config.yaml` (epochs, batch size, learning rate, max pairs). A GPU (e.g.
Colab) is roughly 20x faster.

**Other models.** Changing `embedding.model_name` is enough, e.g.
`sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (same size, trained for
short-query→passage search) or `all-mpnet-base-v2` (set `dimension: 768`,
slower but stronger). Always compare with `evaluate_search.py --compare`.

## 25. FastAPI Usage
```bash
python -m uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000`. The model and FAISS index load once at
startup (see the console log), not on every request.

**`GET /`** — serves the search page.

**`POST /search`**
```json
// request
{ "query": "a sad song about losing someone you love", "top_k": 10 }

// response
{
  "query": "a sad song about losing someone you love",
  "results": [
    {
      "rank": 1,
      "artist": "Artist Name",
      "song": "Song Name",
      "link": "/a/artist/song_12345.html",
      "similarity_score": 0.86,
      "scores": { "best_chunk": 0.83, "whole_song": 0.88, "title": 0.79 },
      "matched_lyrics": "the lyric lines that matched best",
      "lyrics_excerpt": "..."
    }
  ]
}
```
Validated error cases: empty query → 400, `top_k < 1` or missing `query` →
422 (Pydantic), `top_k > 50` → 400, index/model not loaded → 503.

## 26. Example Searches
- "a sad song about lost love"
- "a song about falling in love"
- "lonely and missing someone"
- "a happy song about friendship"
- "a song about heartbreak after a breakup"
- "a romantic song about love"
- "a song about feeling confident and powerful"
- "a peaceful and emotional song"

## 27. Evaluation
```bash
python scripts/evaluate_search.py --dump-only     # writes out/eval_candidates.csv for labelling
python scripts/evaluate_search.py --compare --eval-file configs/eval_queries_pooled.yaml
python scripts/evaluate_search.py --baseline      # adds a TF-IDF keyword baseline
python scripts/evaluate_search.py --weights 0.2,0.65,0.15   # try other chunk,mean,title weights
```
Metrics: Precision@10, Recall@10, MRR, nDCG@10, plus the mean similarity
of the #1 result and the top-10.

**Why there are two label files.** `eval_queries.yaml` was labelled by reading
model_v1's own top-10, so v1 gets 100% recall on it by construction and any
other method looks worse even when its results are good.
`eval_queries_pooled.yaml` pools the top-10 of several methods per query and
judges every song without knowing which method found it (206 judgements). Use the
pooled file to compare methods.

**Results** (pooled labels; test subset of 2,063 songs: each query's top-150 under v1,
every labelled song, and 800 random songs):

| Method | P@10 | nDCG@10 | MRR | sim of #1 |
|---|---|---|---|---|
| TF-IDF keyword baseline | 0.444 | 0.482 | 0.796 | – |
| model_v1 (mean vector) | 0.900 | 0.913 | 1.000 | 0.666 |
| best chunk only | 0.633 | 0.656 | 0.806 | 0.720 |
| **model_v2 (blend + query expansion)** | **0.911** | **0.927** | **1.000** | **0.681** |

The ranking changes give a small, real gain: higher similarity scores and
slightly better precision. With only 9 queries, a difference this small is
within noise. The biggest remaining lever is the embedding model itself
(fine-tuning, section 24). Re-run the table after retraining on the full
dataset; full-dataset numbers will differ from this subset.

## 28. Limitations
- Lyrics-only: no audio, no tempo/BPM/energy/danceability/acousticness —
  the system cannot know how a song sounds.
- No genre labels in the source data — genre-flavored queries are answered
  through lyrical content only, not verified genre membership.
- Short, highly repetitive chunks can score well on short queries, which is
  why the chunk signal has a modest weight.
- A small, single-annotator evaluation set is not a rigorous benchmark.
- Non-English lyrics in the dataset may embed poorly relative to
  English-language queries, since the model is primarily English-trained.

## 29. Future Improvements
- Label more queries (aim for 30+) in `eval_queries_pooled.yaml` so weight
  tuning isn't based on 9 queries.
- Language detection to filter or separately handle non-English lyrics.
- Deduping near-identical lyric content (covers, live versions, e.g. three
  "All Is Loneliness" versions) beyond exact `(artist, song)` matches.
- Enable and tune the cross-encoder re-ranker (`rerank:` in config.yaml).

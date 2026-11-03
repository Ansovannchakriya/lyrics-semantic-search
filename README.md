# Lyrics Semantic Search

Search songs by what their lyrics mean, not by exact words.

Type something like "a sad song about losing someone you love" and the app returns songs whose lyrics are closest in meaning, even if they don't contain any of the words you typed.

It uses the Spotify Million Song Dataset from Kaggle (57,648 songs). This project isn't affiliated with Spotify.

## How it works

1. Each song's lyrics are split into chunks of 5 lines (1 line overlap).
2. Every chunk, every song title, and the average of each song's chunks are embedded with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions).
3. The vectors are stored in three FAISS indexes: chunks, songs and titles.
4. At search time, each song gets a score that blends three cosine similarities:

   ```
   score = 0.65 × whole song + 0.20 × best chunk + 0.15 × title
   ```

5. Query expansion: the query vector is moved slightly toward its top 5 results, and the search runs again.
6. A cross-encoder ([ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)) re-reads the top 50 results together with the query and re-orders them.

The web page shows a match % for each song, a Strong / Good / Weak label, and the lyric lines that matched best.

Note on scores: with MiniLM, a short query compared with lyrics rarely goes above 0.8 cosine, even for a perfect match. Anything above 0.6 is a strong result. The match % rescales this (0.30 → 0%, 0.80 → 100%) and mixes in the re-ranker's score.

## Results

Tested on 9 queries with pooled relevance labels (`configs/eval_queries_pooled.yaml`), on a 2,063-song subset:

| Method | P@10 | nDCG@10 | MRR |
|---|---|---|---|
| TF-IDF keyword search | 0.444 | 0.482 | 0.796 |
| Mean song vector only (v1) | 0.900 | 0.913 | 1.000 |
| Best chunk only | 0.633 | 0.656 | 0.806 |
| Blended score + query expansion (v2) | 0.911 | 0.927 | 1.000 |

v2 is slightly better than v1, but with 9 queries the difference is small. Scoring by the best chunk alone does worse, because short repetitive lines ("love love love") get high scores.

The labels in `configs/eval_queries.yaml` came from v1's own top 10, so v1 always looks best on that file. The pooled file is fairer: the top 10 from several methods was combined and each song was judged without knowing which method found it.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/Ansovannchakriya/lyrics-semantic-search.git
cd lyrics-semantic-search
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS / Linux
python -m pip install -r requirements.txt
```

Download `spotify_millsongdata.csv` from Kaggle (search "Spotify Million Song Dataset") and put it in `data/raw/`.

## Build the index

```bash
python scripts/prepare_dataset.py       # clean + dedupe -> data/processed/songs.csv
python scripts/generate_embeddings.py   # slowest step, about 30-60 min on CPU
python scripts/build_index.py           # -> out/model_v2/
```

To test quickly on 2,000 songs first, run `python scripts/generate_embeddings.py --limit 2000`.

## Run the app

```bash
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The first start downloads the re-ranker model (about 90 MB).

On Windows, use `python -m uvicorn` and `python -m pip` instead of `uvicorn` and `pip`. If the project folder is moved, the `.exe` launchers inside `.venv` break, but `python -m` still works.

### API

`POST /search`

```json
{ "query": "a sad song about losing someone you love", "top_k": 10 }
```

Each result has `artist`, `song`, `link`, `match_percent`, `match_label`, `similarity_score` (raw cosine), `relevance` (re-ranker, 0-1), `scores` (best chunk / whole song / title), `matched_lyrics` and `lyrics_excerpt`.

## Fine-tuning (optional)

`scripts/finetune_model.py` fine-tunes MiniLM on the lyrics without any manual labels. It builds training pairs from the data:

- song title → a chunk of that song's lyrics
- most repeated line (usually the chorus) → another chunk of the same song

It trains with Multiple Negatives Ranking Loss: in each batch, the model has to pick the right chunk for each title or line out of all the chunks in the batch. Songs used in evaluation are left out of training. It prints validation accuracy before and after.

```bash
python scripts/finetune_model.py --max-pairs 5000   # quick run
python scripts/finetune_model.py                    # full run (60k pairs)
```

Then set `embedding.model_name: "out/finetuned_model"` in `configs/config.yaml` and run `generate_embeddings.py` and `build_index.py` again. A GPU (e.g. Colab) is much faster.

## Evaluation

```bash
python scripts/evaluate_search.py --compare --eval-file configs/eval_queries_pooled.yaml
python scripts/evaluate_search.py --no-rerank --eval-file configs/eval_queries_pooled.yaml
python scripts/evaluate_search.py --baseline                 # adds TF-IDF
python scripts/evaluate_search.py --weights 0.2,0.65,0.15    # try other weights (chunk, mean, title)
python scripts/evaluate_search.py --dump-only                # export results to label by hand
```

Reports Precision@10, Recall@10, MRR, nDCG@10 and average scores.

## Configuration

Everything is in `configs/config.yaml`:

- `embedding.model_name`: any sentence-transformers model, or a local folder
- `lyrics`: chunk size and overlap
- `search.weights`: chunk / mean / title weights
- `search.query_expansion`: on/off, number of feedback songs, strength
- `rerank`: on/off, model, number of candidates, weight
- `display`: match % range and label thresholds
- `finetune`: epochs, batch size, learning rate, number of pairs

## Project structure

```
lyrics-semantic-search/
├── app/
│   ├── main.py               # FastAPI routes
│   ├── search_engine.py      # loads indexes + models, runs search
│   ├── ranking.py            # scoring, query expansion, match %
│   ├── templates/index.html
│   └── static/               # script.js, style.css
├── configs/
│   ├── config.yaml
│   ├── eval_queries.yaml
│   └── eval_queries_pooled.yaml
├── scripts/
│   ├── prepare_dataset.py
│   ├── text_chunking.py
│   ├── generate_embeddings.py
│   ├── build_index.py
│   ├── finetune_model.py
│   └── evaluate_search.py
├── data/                     # not in git
├── out/                      # not in git
└── requirements.txt
```

The dataset, embeddings, FAISS indexes and models are not in the repo because they're too large (hundreds of MB). The scripts above rebuild them.

## Limitations

- Lyrics only. There's no audio, tempo or genre data, so it can't tell how a song sounds.
- The model is trained mostly on English, so non-English lyrics match poorly.
- Duplicate versions of the same song (live, album, etc.) can all show up in the results.
- 9 evaluation queries is a small test set. More labeled queries would give more reliable numbers.

## Author

Sovannchakriya AN ([@Ansovannchakriya](https://github.com/Ansovannchakriya))

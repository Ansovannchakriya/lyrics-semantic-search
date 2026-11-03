"""Step 2: embed every song.

model_v2 saves THREE kinds of vectors instead of one:
  * chunk_embeddings.npy  - one vector per lyric chunk (float16 to save disk/RAM).
                            Search scores a song by its BEST chunk, so a strong
                            chorus is no longer diluted by neutral verses.
  * embeddings.npy        - one mean-pooled vector per song (exactly like v1).
  * title_embeddings.npy  - one vector per song title (titles often name the theme:
                            "Loneliness", "Missing You", "Celebration").
plus chunk_offsets.npy: song i owns chunk rows [offsets[i], offsets[i+1]).

Usage:
    python scripts/generate_embeddings.py
    python scripts/generate_embeddings.py --limit 2000   # quick smoke test on 2000 songs
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from text_chunking import lyrics_to_chunks  # noqa: E402


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(emb_cfg: dict):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers is not installed.")
        print("  Run: pip install -r requirements.txt")
        sys.exit(1)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"Loading model: {emb_cfg['model_name']}  (device: {device})")
    model = SentenceTransformer(emb_cfg["model_name"], device=device)
    if emb_cfg.get("max_seq_length"):
        model.max_seq_length = emb_cfg["max_seq_length"]
    return model


def encode(model, texts, emb_cfg, prefix=""):
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(
        texts,
        batch_size=emb_cfg["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only embed the first N songs (for a quick test run)")
    args = parser.parse_args()

    config = load_config()
    emb_cfg = config["embedding"]
    lyr_cfg = config["lyrics"]
    data_cfg = config["data"]

    processed_path = Path(data_cfg["processed_path"])
    if not processed_path.exists():
        print(f"ERROR: {processed_path} not found. Run prepare_dataset.py first.")
        sys.exit(1)
    Path(data_cfg["embeddings_path"]).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned dataset from {processed_path} ...")
    df = pd.read_csv(processed_path)
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"  {len(df)} songs")

    model = load_model(emb_cfg)
    dimension = model.get_sentence_embedding_dimension()
    if dimension != emb_cfg["dimension"]:
        print(f"ERROR: model produces {dimension}-dim vectors but config.yaml says "
              f"{emb_cfg['dimension']}. Update embedding.dimension in config.yaml.")
        sys.exit(1)

    # 1. Chunk every song, remembering where each song's chunks start/end.
    print(f"Chunking lyrics ({lyr_cfg['sentences_per_chunk']} lines per chunk, "
          f"{lyr_cfg['sentence_overlap']} overlap) ...")
    all_chunks, offsets = [], [0]
    for text in df["text"]:
        all_chunks.extend(lyrics_to_chunks(text, lyr_cfg["sentences_per_chunk"],
                                           lyr_cfg["sentence_overlap"]))
        offsets.append(len(all_chunks))
    offsets = np.asarray(offsets, dtype=np.int64)
    print(f"  {len(all_chunks)} chunks (avg {len(all_chunks) / len(df):.1f} per song)")

    # 2. Encode chunks (the slow step).
    print("Encoding lyric chunks (slow step) ...")
    t0 = time.time()
    chunk_emb = encode(model, all_chunks, emb_cfg, emb_cfg.get("passage_prefix", ""))
    print(f"  done in {time.time() - t0:.0f}s")

    # 3. Mean-pool per song (v1 vector), then L2-normalise.
    song_emb = np.zeros((len(df), dimension), dtype=np.float32)
    for i in range(len(df)):
        v = chunk_emb[offsets[i]:offsets[i + 1]].mean(axis=0)
        n = np.linalg.norm(v)
        song_emb[i] = v / n if n > 0 else v

    # 4. Encode titles.
    print("Encoding song titles ...")
    title_emb = encode(model, df["song"].astype(str).tolist(), emb_cfg,
                       emb_cfg.get("passage_prefix", ""))

    np.save(data_cfg["chunk_embeddings_path"], chunk_emb.astype(np.float16))
    np.save(data_cfg["chunk_offsets_path"], offsets)
    np.save(data_cfg["embeddings_path"], song_emb)
    np.save(data_cfg["title_embeddings_path"], title_emb)
    print(f"Saved chunk embeddings {chunk_emb.shape}, song embeddings {song_emb.shape}, "
          f"title embeddings {title_emb.shape}")

    # 5. Metadata in the SAME row order. Full lyrics are kept so the app can
    #    show the exact chunk that matched the query.
    metadata = df[["artist", "song", "link", "text"]].copy()
    metadata["lyrics_excerpt"] = metadata["text"].str.replace("\n", " ").str.slice(0, 220)
    metadata.to_csv(data_cfg["metadata_path"], index=False)
    print(f"Saved metadata: {len(metadata)} rows -> {data_cfg['metadata_path']}")
    print("\nDone. Next: python scripts/build_index.py")


if __name__ == "__main__":
    main()

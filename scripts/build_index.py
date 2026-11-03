"""Step 3: build the three FAISS indexes used by model_v2.

  chunks.index  - every lyric chunk (float16 scalar-quantised flat index:
                  exact search, half the memory of float32)
  songs.index   - one mean-pooled vector per song (same as model_v1)
  titles.index  - one vector per song title

All vectors are L2-normalised, so inner product == cosine similarity.
"""
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_vectors(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        print(f"ERROR: {path} not found. Run generate_embeddings.py first.")
        sys.exit(1)
    v = np.load(path).astype(np.float32)
    faiss.normalize_L2(v)
    print(f"  {name}: {v.shape}")
    return v


def main():
    config = load_config()
    d, o = config["data"], config["output"]
    dim = config["embedding"]["dimension"]
    Path(o["model_dir"]).mkdir(parents=True, exist_ok=True)

    print("Loading embeddings ...")
    chunks = load_vectors(Path(d["chunk_embeddings_path"]), "chunk embeddings")
    songs = load_vectors(Path(d["embeddings_path"]), "song embeddings")
    titles = load_vectors(Path(d["title_embeddings_path"]), "title embeddings")
    offsets = np.load(d["chunk_offsets_path"])
    metadata = pd.read_csv(d["metadata_path"])

    # integrity checks
    n = len(metadata)
    problems = []
    if songs.shape[0] != n:
        problems.append(f"song embeddings ({songs.shape[0]}) != metadata rows ({n})")
    if titles.shape[0] != n:
        problems.append(f"title embeddings ({titles.shape[0]}) != metadata rows ({n})")
    if len(offsets) != n + 1 or offsets[-1] != chunks.shape[0]:
        problems.append("chunk_offsets.npy does not match chunk_embeddings.npy")
    for name, arr in [("chunk", chunks), ("song", songs), ("title", titles)]:
        if arr.shape[1] != dim:
            problems.append(f"{name} embeddings are {arr.shape[1]}-dim, config says {dim}")
    if problems:
        print("ERROR:\n  " + "\n  ".join(problems) + "\nRe-run generate_embeddings.py.")
        sys.exit(1)

    print("Building indexes ...")
    chunk_index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16,
                                             faiss.METRIC_INNER_PRODUCT)
    chunk_index.train(chunks)  # no-op for fp16, but required by the API
    chunk_index.add(chunks)
    song_index = faiss.IndexFlatIP(dim)
    song_index.add(songs)
    title_index = faiss.IndexFlatIP(dim)
    title_index.add(titles)
    print(f"  chunks: {chunk_index.ntotal}   songs: {song_index.ntotal}   "
          f"titles: {title_index.ntotal}")

    faiss.write_index(chunk_index, o["chunk_index_path"])
    faiss.write_index(song_index, o["song_index_path"])
    faiss.write_index(title_index, o["title_index_path"])
    np.save(o["chunk_offsets_path"], offsets)
    metadata.to_csv(o["metadata_path"], index=False)
    with open(o["config_used_path"], "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Saved everything to {o['model_dir']}/")
    print("\nDone. Start the app:  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()

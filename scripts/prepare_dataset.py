import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REQUIRED_COLUMNS = ["artist", "song", "link", "text"]


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Whitespace/structural cleanup. Deliberately NOT stemming, NOT
    removing stopwords, NOT lowercasing aggressively -- the semantic
    meaning of the lyrics needs to stay intact for the embedding model.

    Important: line breaks are KEPT (only \\r\\n normalized to \\n), not
    collapsed into spaces. Lyrics are naturally broken into short phrase
    units by line, and scripts/text_chunking.py's sentence splitter relies
    on that structure as its primary signal -- collapsing lines here would
    silently turn every song into one giant run-on "sentence" downstream.
    """
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # strip structural tags like [Chorus], [Verse 2] -- formatting noise,
    # not lyrical content
    t = re.sub(r"\[[^\]]*\]", " ", t)
    # collapse horizontal whitespace only, per line; keep the newlines
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def main():
    config = load_config()
    raw_path = Path(config["data"]["raw_path"])
    processed_path = Path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from {raw_path} ...")
    if not raw_path.exists():
        print(f"ERROR: raw dataset not found at {raw_path}")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    original_rows = len(df)
    print(f"  Loaded {original_rows} rows")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: dataset is missing required columns: {missing_cols}")
        print(f"  Found columns: {list(df.columns)}")
        sys.exit(1)
    df = df[REQUIRED_COLUMNS].copy()

    # remove rows with missing artist / song / lyrics 
    before = len(df)
    df = df.dropna(subset=["artist", "song", "text"])
    dropped_missing = before - len(df)

    # clean whitespace / structural noise in lyrics
    df["text"] = df["text"].apply(clean_text)
    df["artist"] = df["artist"].astype(str).str.strip()
    df["song"] = df["song"].astype(str).str.strip()

    # drop empty lyrics after cleaning 
    before = len(df)
    df = df[df["text"].str.len() > 0]
    dropped_empty = before - len(df)

    # remove duplicate songs (same artist + song) 
    before = len(df)
    df = df.drop_duplicates(subset=["artist", "song"], keep="first")
    dropped_duplicates = before - len(df)

    df = df.reset_index(drop=True)

    lyric_lengths = df["text"].str.len()

    print("\n--- Preprocessing statistics -----------------------------")
    print(f"Original rows:              {original_rows}")
    print(f"Removed (missing fields):   {dropped_missing}")
    print(f"Removed (empty lyrics):     {dropped_empty}")
    print(f"Removed (duplicate songs):  {dropped_duplicates}")
    print(f"Rows after cleaning:        {len(df)}")
    print(f"Average lyric length (chars): {lyric_lengths.mean():.1f}")
    print(f"Maximum lyric length (chars): {lyric_lengths.max()}")
    print("------------------------------------------------------------\n")

    df.to_csv(processed_path, index=False)
    print(f"Saved cleaned dataset to {processed_path}")


if __name__ == "__main__":
    main()

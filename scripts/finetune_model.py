"""Optional step: fine-tune the sentence-transformer on YOUR lyrics dataset.
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from text_chunking import lyrics_to_chunks, split_into_sentences  # noqa: E402


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- data ---- #
def hook_line(text: str) -> str | None:
    """Most repeated lyric line with >= 3 words (a cheap chorus detector)."""
    lines = [ln.strip() for ln in split_into_sentences(text) if len(ln.split()) >= 3]
    if not lines:
        return None
    line, count = Counter(ln.lower() for ln in lines).most_common(1)[0]
    if count < 2:
        return None
    return next(ln for ln in lines if ln.lower() == line)


def build_pairs(df, lyr_cfg, pairs_per_song, rng):
    pairs = []
    for title, text in zip(df["song"].astype(str), df["text"]):
        chunks = lyrics_to_chunks(text, lyr_cfg["sentences_per_chunk"],
                                  lyr_cfg["sentence_overlap"])
        chunks = [c for c in chunks if len(c.split()) >= 8]
        if not chunks:
            continue
        song_pairs = []
        if len(title) >= 3:
            song_pairs.append((title, rng.choice(chunks)))
        hook = hook_line(text)
        if hook:
            others = [c for c in chunks if hook.lower() not in c.lower()] or chunks
            song_pairs.append((hook, rng.choice(others)))
        pairs.extend(song_pairs[:pairs_per_song])
    return pairs


def make_batches(pairs, batch_size, rng):
    """Shuffle into batches with no duplicate anchors inside a batch
    (two songs both called "Love" in one batch would be false negatives)."""
    pool = pairs[:]
    rng.shuffle(pool)
    batches, cur, seen, leftover = [], [], set(), []
    for p in pool:
        key = p[0].lower()
        if key in seen:
            leftover.append(p)
            continue
        cur.append(p)
        seen.add(key)
        if len(cur) == batch_size:
            batches.append(cur)
            cur, seen = [], set()
            # retry postponed pairs in the next batch
            pool_tail, leftover = leftover, []
            for q in pool_tail:
                if q[0].lower() not in seen and len(cur) < batch_size:
                    cur.append(q)
                    seen.add(q[0].lower())
                else:
                    leftover.append(q)
    if len(cur) >= 2:
        batches.append(cur)
    return batches


# --------------------------------------------------------------- model ---- #
def tokenize(model, texts):
    """Works across sentence-transformers versions (tokenize / preprocess)."""
    for obj, name in [(model, "tokenize"), (model, "preprocess"), (model[0], "tokenize")]:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(texts)
    raise RuntimeError("Could not find a tokenize() method on this sentence-transformers version")


def embed(model, texts, device):
    import torch.nn.functional as F
    features = tokenize(model, texts)
    features = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in features.items()}
    out = model(features)["sentence_embedding"]
    return F.normalize(out, dim=-1)


def retrieval_accuracy(model, pairs):
    """Top-1 accuracy: for each anchor, is its own chunk the best match among
    all chunks in the held-out set? Used to show training actually helped."""
    if not pairs:
        return float("nan")
    a = model.encode([p[0] for p in pairs], normalize_embeddings=True, batch_size=64)
    b = model.encode([p[1] for p in pairs], normalize_embeddings=True, batch_size=64)
    return float(np.mean(np.argmax(a @ b.T, axis=1) == np.arange(len(pairs))))


def main():
    cfg = load_config()
    ft = cfg["finetune"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=ft["epochs"])
    ap.add_argument("--batch-size", type=int, default=ft["batch_size"])
    ap.add_argument("--lr", type=float, default=ft["learning_rate"])
    ap.add_argument("--max-pairs", type=int, default=ft["max_pairs"])
    ap.add_argument("--base-model", default=ft["base_model"])
    ap.add_argument("--output-dir", default=ft["output_dir"])
    args = ap.parse_args()

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: needs torch + sentence-transformers (pip install -r requirements.txt)")
        sys.exit(1)

    seed = ft.get("seed", 42)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(cfg["data"]["processed_path"])
    print(f"Loaded {len(df)} songs")

    if ft.get("exclude_eval_songs", True) and Path("configs/eval_queries.yaml").exists():
        with open("configs/eval_queries.yaml") as f:
            ev = yaml.safe_load(f)["queries"]
        held = {tuple(p) for q in ev for p in q.get("relevant_songs", [])}
        before = len(df)
        df = df[[(a, s) not in held for a, s in zip(df["artist"], df["song"])]]
        print(f"Excluded {before - len(df)} evaluation songs from training (no leakage)")

    # split songs 98% train / 2% validation BEFORE building pairs
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n_val = max(200, len(df) // 50)
    val_df, train_df = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]

    train_pairs = build_pairs(train_df, cfg["lyrics"], ft["pairs_per_song"], rng)
    rng.shuffle(train_pairs)
    train_pairs = train_pairs[: args.max_pairs]
    val_pairs = build_pairs(val_df, cfg["lyrics"], 1, rng)[:500]
    print(f"Training pairs: {len(train_pairs)}   validation pairs: {len(val_pairs)}")
    print("Example pairs:")
    for a, b in train_pairs[:3]:
        print(f"  [{a}]  ->  {b[:90]}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading base model {args.base_model} on {device}")
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = 128  # chunks are short; shorter = faster training

    model.eval()
    acc_before = retrieval_accuracy(model, val_pairs)
    print(f"Validation top-1 accuracy BEFORE fine-tuning: {acc_before:.3f}")

    batches_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    total_steps = batches_per_epoch * args.epochs
    warmup = int(total_steps * ft.get("warmup_ratio", 0.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    scale = 20.0  # temperature: cosine * 20 -> logits

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        recent = []
        batches = make_batches(train_pairs, args.batch_size, rng)
        for b, batch in enumerate(batches, start=1):
            a = embed(model, [p[0] for p in batch], device)
            p = embed(model, [p[1] for p in batch], device)
            logits = a @ p.T * scale
            labels = torch.arange(len(batch), device=device)
            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            recent = (recent + [loss.item()])[-50:]
            if b % 50 == 0 or b == len(batches):
                el = time.time() - t0
                eta = el / step * (total_steps - step)
                print(f"  epoch {epoch + 1}  batch {b}/{len(batches)}  "
                      f"loss {sum(recent) / len(recent):.4f}  "
                      f"elapsed {el / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)

    model.eval()
    acc_after = retrieval_accuracy(model, val_pairs)
    print(f"\nValidation top-1 accuracy: before {acc_before:.3f}  ->  after {acc_after:.3f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    with open(out / "training_info.json", "w") as f:
        json.dump({"base_model": args.base_model, "epochs": args.epochs,
                   "batch_size": args.batch_size, "lr": args.lr,
                   "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
                   "val_top1_before": acc_before, "val_top1_after": acc_after}, f, indent=2)
    print(f"Saved fine-tuned model -> {out}")
    print("\nNext: set embedding.model_name to \"out/finetuned_model\" in configs/config.yaml,")
    print("then run generate_embeddings.py and build_index.py again.")
    if acc_after <= acc_before:
        print("NOTE: validation accuracy did not improve -- try a lower --lr or more pairs, "
              "or keep the base model.")


if __name__ == "__main__":
    main()

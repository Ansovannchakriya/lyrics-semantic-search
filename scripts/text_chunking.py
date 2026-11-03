import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LONG_LINE_CHARS = 140  # a line longer than this probably holds >1 sentence


def split_into_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) <= 1:
        # No line breaks survived (already flattened) - fall back to
        # straight punctuation splitting on the whole text.
        lines = [text]

    sentences = []
    for line in lines:
        if len(line) > _LONG_LINE_CHARS and _SENTENCE_END.search(line):
            parts = _SENTENCE_END.split(line)
            sentences.extend(p.strip() for p in parts if p.strip())
        else:
            sentences.append(line)

    return sentences if sentences else ([text.strip()] if text.strip() else [])


def chunk_sentences(
    sentences: list[str],
    sentences_per_chunk: int = 5,
    sentence_overlap: int = 1,
) -> list[str]:
    """Group a sentence list into overlapping chunks, then join each
    chunk's sentences back into one string ready to hand to the encoder.
    """
    if not sentences:
        return [""]

    step = max(1, sentences_per_chunk - sentence_overlap)
    chunks = []
    for start in range(0, len(sentences), step):
        window = sentences[start:start + sentences_per_chunk]
        if window:
            chunks.append(" ".join(window))
        if start + sentences_per_chunk >= len(sentences):
            break
    return chunks if chunks else [" ".join(sentences)]


def lyrics_to_chunks(
    text: str,
    sentences_per_chunk: int = 5,
    sentence_overlap: int = 1,
) -> list[str]:
    sentences = split_into_sentences(text)
    return chunk_sentences(sentences, sentences_per_chunk, sentence_overlap)

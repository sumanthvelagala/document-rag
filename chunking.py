# universal sliding window chunker — no domain assumptions, works on any document
def sliding_window_chunks(text, token_limit=128, overlap=32):
    words = text.split()
    chunks = []
    start = 0
    step = token_limit - overlap

    while start < len(words):
        chunk_words = words[start:start + token_limit]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        start += step

    return chunks





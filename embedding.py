from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

def tokenize(text):
    return model.encode(text, normalize_embeddings=True)

def embeddings(chunks):
    embedded_chunks = []
    for title, chunk in chunks:
        embedding = tokenize(chunk)
        embedded_chunks.append((title, chunk, embedding))
    return embedded_chunks

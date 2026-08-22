import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from embedding import tokenize
from rank_bm25 import BM25Okapi

class QueryInput(BaseModel):
    query: str
class ChunkInput(BaseModel):
    title: str
    chunk: str
class StoreInput(BaseModel):
    pdf_id: str
    chunks: list[ChunkInput]


app = FastAPI()

client = chromadb.PersistentClient(path="/Users/Sumanth/Terminal/DataBase")

# BM25 index — built lazily, invalidated on every /store
_bm25 = None
_bm25_docs: list = []
_bm25_metas: list = []
_bm25_ids: list = []


def _rebuild_bm25():
    global _bm25, _bm25_docs, _bm25_metas, _bm25_ids
    col = client.get_or_create_collection("rag_documents")
    data = col.get()
    _bm25_docs  = data["documents"]
    _bm25_metas = data["metadatas"]
    _bm25_ids   = data["ids"]
    _bm25 = BM25Okapi([doc.lower().split() for doc in _bm25_docs])


@app.post("/store")
def store_chunks(input: StoreInput):
    global _bm25
    pdf_id = input.pdf_id
    collection = client.get_or_create_collection(name="rag_documents")

    for i, chunk in enumerate(input.chunks):
        embedding = tokenize(chunk.chunk)
        collection.add(
            ids=[f"{pdf_id}_chunk{i}"],
            documents=[chunk.chunk],
            embeddings=[embedding.tolist()],
            metadatas=[{"title": chunk.title, "pdf_id": pdf_id}]
        )

    _bm25 = None  # invalidate so next query rebuilds with new docs
    return {"message": "Chunks embedded and stored successfully."}


@app.post("/query")
def handle_query(input: QueryInput):
    global _bm25
    if _bm25 is None:
        _rebuild_bm25()

    collection = client.get_or_create_collection(name="rag_documents")

    # --- dense retrieval: top 50 candidates ---
    query_embedding = tokenize(input.query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=50
    )
    dense_ids   = results["ids"][0]
    dense_docs  = results["documents"][0]
    dense_metas = results["metadatas"][0]

    # --- BM25 retrieval: top 50 candidates ---
    tokens = input.query.lower().split()
    bm25_scores = _bm25.get_scores(tokens)
    top50_bm25  = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:50]

    # --- Reciprocal Rank Fusion ---
    RRF_K = 60
    fused: dict = {}

    for rank, (doc_id, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
        if doc_id not in fused:
            fused[doc_id] = {"score": 0.0, "doc": doc, "meta": meta}
        fused[doc_id]["score"] += 2.0 / (RRF_K + rank + 1)  # dense weighted 2x

    for rank, idx in enumerate(top50_bm25):
        doc_id = _bm25_ids[idx]
        if doc_id not in fused:
            fused[doc_id] = {"score": 0.0, "doc": _bm25_docs[idx], "meta": _bm25_metas[idx]}
        fused[doc_id]["score"] += 1.0 / (RRF_K + rank + 1)

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:10]

    return {"results": [
        {
            "pdf_id": item["meta"].get("pdf_id", "unknown_pdf_id"),
            "title":  item["meta"].get("title",  "unknown_title"),
            "chunk":  item["doc"],
        }
        for item in ranked
    ]}

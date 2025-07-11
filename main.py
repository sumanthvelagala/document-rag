
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from embedding import tokenize,embeddings

class QueryInput(BaseModel):
    query: str
class ChunkInput(BaseModel):
    title: str
    chunk: str
class StoreInput(BaseModel):
    pdf_id: str
    chunks: list[ChunkInput]


app = FastAPI()


# storage to DB
client = chromadb.PersistentClient(path="/Users/Sumanth/Terminal/DataBase")

@app.post("/store")
def store_chunks(input: StoreInput):
    pdf_id = input.pdf_id
    collection = client.get_or_create_collection(name="scientific_paper_RAG")

    for i, chunk in enumerate(input.chunks):
        embedding = tokenize(chunk.chunk)
        collection.add(
            ids=[f"{pdf_id}_chunk{i}"],
            documents=[chunk.chunk],
            embeddings=[embedding.tolist()],
            metadatas=[{"title": chunk.title,
                        "pdf_id": pdf_id }]
        )

    return {"message": "Chunks embedded and stored successfully."}

@app.post("/query")
def handle_query(input:QueryInput):
    query_embedding = tokenize(input.query)
    collections = client.get_or_create_collection(name="scientific_paper_RAG")
    results = collections.query(query_embeddings = [query_embedding.tolist()],
    n_results = 5
    )
    response = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        response.append({
        "pdf_id": metadata.get("pdf_id", "unknown_pdf_id"),
        "title": metadata.get("title", "unknown_title"),
        "chunk": doc
    })
    
    return {"results": response}


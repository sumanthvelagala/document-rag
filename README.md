---
title: Document RAG
emoji: 📄
colorFrom: gray
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# Document RAG — Hybrid Dense + BM25 Retrieval

Upload any PDF and ask questions. Uses hybrid retrieval (dense + BM25 with Reciprocal Rank Fusion) for 91.7% Recall@10 across 60 benchmark questions spanning 15 papers and 6 domains.

## Architecture

```mermaid
flowchart TD
    A([PDF Upload]) --> B[pypdf\ntext extraction]
    B --> C[clean_text\nfix hyphens · strip page numbers]
    C --> D[Sliding Window Chunking\n128 words · 32 overlap]
    D --> E[all-mpnet-base-v2\n768-dim embeddings]
    D --> G[(BM25 Index)]
    E --> F[(ChromaDB\nin-memory · per session)]

    Q([User Query]) --> QE[Encode Query\nall-mpnet-base-v2]
    Q --> QB[Tokenize Query\nBM25]

    QE --> DR[Dense Retrieval\ntop-50 by cosine similarity]
    QB --> BR[BM25 Retrieval\ntop-50 by keyword score]
    F --> DR
    G --> BR

    DR --> RRF[RRF Fusion\nscore = 2·dense + 1·BM25]
    BR --> RRF
    RRF --> TOP[Top-10 Chunks]
    TOP --> LLM[Qwen2.5-7B-Instruct\nHF Inference API]
    LLM --> ANS([Answer])
```

> **Privacy:** PDFs are never stored on disk. Text is extracted server-side, chunked and embedded into RAM, then the original file is discarded. All session data (chunks, embeddings, BM25 index) lives in server memory and is wiped when the session ends.

### Browser ↔ Server Flow

```mermaid
sequenceDiagram
    participant B as Browser
    participant S as HF Spaces Server (RAM)
    participant H as HF Inference API

    B->>S: Upload PDF bytes
    S->>S: pypdf extract → clean → chunk → embed
    S->>S: Store chunks in ChromaDB + BM25 index
    S->>S: Discard original PDF
    S-->>B: "X chunks added"

    B->>S: Submit query
    S->>S: Encode query (all-mpnet-base-v2)
    S->>S: Dense retrieval top-50 + BM25 top-50
    S->>S: RRF fusion → top-10 chunks
    S->>H: top-10 chunks + question
    H-->>S: Generated answer (Qwen2.5-7B)
    S-->>B: Answer + retrieved chunks
```

## Eval Results (60 questions, 15 papers, 6 domains)

| Metric | Hybrid RRF | BM25 baseline |
|---|---|---|
| Recall@10 | **91.7% (55/60)** | 68.3% (41/60) |
| Outperforms BM25 by | **+34%** | — |
| Avg warm latency | **34ms** | <1ms |

## Accuracy Journey

| Step | Change | Accuracy |
|---|---|---|
| Baseline | SciBERT CLS embeddings | 0% |
| Swap model | all-MiniLM-L6-v2 | 55% |
| Better model | all-mpnet-base-v2 | 65% |
| Smaller chunks | 200 → 128 tokens | 70% |
| Hybrid RRF (1:1) | dense + BM25 fusion | 85% |
| Weighted RRF (2:1) | dense 2x weight | 90% |
| Universal chunker | remove title heuristic | **91.7%** |

## Models

- **Embeddings**: `all-mpnet-base-v2` (sentence-transformers) — 768-dim, retrieval-optimized
- **Generation**: `Qwen/Qwen2.5-7B-Instruct` via HF Inference API
- **Vector store**: ChromaDB (in-memory per session for frontend, persistent for eval pipeline)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Gradio app (deployed on HF Spaces, or run locally)
python app.py

# Streamlit app (local development)
streamlit run front.py

# Eval pipeline (requires papers/ directory and FastAPI running)
uvicorn main:app --host 127.0.0.1 --port 8000
python ingest.py --fresh
python eval.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio app — HF Spaces deployment, per-session upload + hybrid search + generation |
| `front.py` | Streamlit app — local development and testing |
| `main.py` | FastAPI backend — `/store` and `/query` endpoints |
| `ingest.py` | Batch ingest papers into persistent ChromaDB for eval |
| `eval.py` | Recall@10 evaluation: hybrid RRF vs BM25 baseline |
| `eval_llm.py` | End-to-end eval: retrieval → generation → cosine similarity |
| `embedding.py` | all-mpnet-base-v2 encode wrapper |
| `chunking.py` | Universal sliding window chunker |
| `cleaners.py` | PDF text cleaning (hyphen breaks, page numbers, whitespace) |

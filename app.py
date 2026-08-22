import os
import uuid
from io import BytesIO
import gradio as gr
import spaces
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from cleaners import clean_text
from chunking import sliding_window_chunks

MAX_CHUNKS    = 5000
CHUNK_TOKENS  = 128
CHUNK_OVERLAP = 32
RETRIEVE_K    = 50
RRF_FINAL_K   = 10
LLM_K         = 5
RRF_K         = 60


# ── cached globals (loaded once, shared across sessions) ──────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-mpnet-base-v2")
    return _embedder

@spaces.GPU
def encode(texts):
    return get_embedder().encode(texts, normalize_embeddings=True)


# ── per-session state ─────────────────────────────────────────────────────────
def init_session():
    import chromadb
    sid = str(uuid.uuid4())[:8]
    client     = chromadb.Client()
    collection = client.create_collection(f"session_{sid}")
    return {
        "uploaded":    [],
        "chunk_count": 0,
        "bm25_docs":   [],
        "bm25_metas":  [],
        "bm25_ids":    [],
        "bm25":        None,
        "collection":  collection,
    }


# ── pdf ingestion ─────────────────────────────────────────────────────────────
def ingest_pdf(file_bytes, filename, state):
    if filename in state["uploaded"]:
        return state, f"{filename}: already uploaded."
    if len(state["uploaded"]) >= 30:
        return state, "30 PDF limit reached."

    raw = ""
    for page in PdfReader(BytesIO(file_bytes)).pages:
        raw += (page.extract_text() or "") + "\n"

    text   = clean_text(raw)
    chunks = sliding_window_chunks(text, CHUNK_TOKENS, CHUNK_OVERLAP)

    if state["chunk_count"] + len(chunks) > MAX_CHUNKS:
        return state, f"{filename}: would exceed {MAX_CHUNKS:,} chunk limit."

    embeddings = encode(chunks)  # one GPU call for all chunks
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{filename}_chunk{i}"
        state["collection"].add(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[emb.tolist()],
            metadatas=[{"pdf_id": filename}],
        )
        state["bm25_docs"].append(chunk)
        state["bm25_metas"].append({"pdf_id": filename})
        state["bm25_ids"].append(chunk_id)

    state["uploaded"].append(filename)
    state["chunk_count"] += len(chunks)
    state["bm25"] = BM25Okapi([d.lower().split() for d in state["bm25_docs"]])
    return state, f"{filename}: {len(chunks)} chunks added."


# ── hybrid retrieval ──────────────────────────────────────────────────────────
def hybrid_search(query, state, filter_pdfs=None):
    if not state["bm25_docs"]:
        return []

    # build active doc set — None or empty means search all
    active = set(filter_pdfs) if filter_pdfs else None

    # filter BM25 pool to active docs
    if active:
        bm25_pool = [(i, state["bm25_ids"][i])
                     for i, m in enumerate(state["bm25_metas"])
                     if m["pdf_id"] in active]
    else:
        bm25_pool = list(enumerate(state["bm25_ids"]))

    k     = min(RETRIEVE_K, len(state["bm25_docs"]))
    q_emb = encode(query)

    # dense retrieval — filter via ChromaDB where clause
    query_kwargs = dict(query_embeddings=[q_emb.tolist()], n_results=min(k, len(bm25_pool) or k))
    if active:
        query_kwargs["where"] = (
            {"pdf_id": list(active)[0]} if len(active) == 1
            else {"pdf_id": {"$in": list(active)}}
        )
    results     = state["collection"].query(**query_kwargs)
    dense_ids   = results["ids"][0]
    dense_docs  = results["documents"][0]
    dense_metas = results["metadatas"][0]

    # BM25 retrieval over active pool only
    scores   = state["bm25"].get_scores(query.lower().split())
    top_bm25 = sorted(bm25_pool, key=lambda t: scores[t[0]], reverse=True)[:k]

    fused = {}
    for rank, (doc_id, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
        if doc_id not in fused:
            fused[doc_id] = {"score": 0.0, "doc": doc, "meta": meta}
        fused[doc_id]["score"] += 2.0 / (RRF_K + rank + 1)

    for rank, (idx, doc_id) in enumerate(top_bm25):
        if doc_id not in fused:
            fused[doc_id] = {
                "score": 0.0,
                "doc":   state["bm25_docs"][idx],
                "meta":  state["bm25_metas"][idx],
            }
        fused[doc_id]["score"] += 1.0 / (RRF_K + rank + 1)

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:RRF_FINAL_K]


# ── answer generation — extractive with chunk-weighted scoring + semantic dedup
def generate_answer(query, chunks):
    import numpy as np

    n_chunks = len(chunks)
    sentences, sources, chunk_weights = [], [], []
    for rank, c in enumerate(chunks):
        w = 1.0 - (rank / n_chunks) * 0.3   # top chunk gets full weight, last gets 0.7x
        for s in c["doc"].split("."):
            s = s.strip()
            if len(s) > 20:
                sentences.append(s)
                sources.append(c["meta"].get("pdf_id", "unknown"))
                chunk_weights.append(w)

    if not sentences:
        return "Not found in the document."

    q_emb    = encode(query)
    s_embs   = encode(sentences)
    sim_scores = s_embs @ q_emb

    if float(sim_scores.max()) < 0.50:
        return "Not found in the document."

    scores = sim_scores * np.array(chunk_weights)

    ranked = scores.argsort()[::-1]
    selected, selected_embs = [], []
    for idx in ranked:
        if len(selected) >= 5:
            break
        emb = s_embs[idx]
        if selected_embs:
            sims = [float(emb @ se) for se in selected_embs]
            if max(sims) > 0.90:
                continue
        selected.append((sentences[idx], sources[idx]))
        selected_embs.append(emb)

    return "\n".join(f"• {s} [{src}]" for s, src in selected)


# ── gradio event handlers ─────────────────────────────────────────────────────
def process_upload(files, state):
    if state is None:
        state = init_session()
    if not files:
        return state, "No files selected.", _doc_list(state), _chunk_info(state), gr.update()

    msgs = []
    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            state, msg = ingest_pdf(f.read(), filename, state)
        msgs.append(msg)

    return state, "\n".join(msgs), _doc_list(state), _chunk_info(state), gr.update(choices=state["uploaded"])


def search_and_generate(query, filter_pdfs, state):
    if state is None or not state["uploaded"]:
        return "Upload at least one PDF first.", "", state
    if not query.strip():
        return "Enter a question.", "", state

    chunks = hybrid_search(query, state, filter_pdfs or None)
    if not chunks:
        return "No results found.", "", state

    answer = generate_answer(query, chunks)
    return answer, _format_chunks(chunks), state


def clear_session(state):
    state = init_session()
    return state, "All documents cleared.", "No documents uploaded yet.", "0 / 5,000 chunks used", gr.update(choices=[], value=[])


def _doc_list(state):
    if not state["uploaded"]:
        return "No documents uploaded yet."
    return "\n".join(f"• {name}" for name in state["uploaded"])

def _chunk_info(state):
    return f"{state['chunk_count']:,} / {MAX_CHUNKS:,} chunks used"

def _format_chunks(chunks):
    out = ""
    for i, c in enumerate(chunks, 1):
        out += f"**#{i} · {c['meta'].get('pdf_id', 'unknown')}**\n\n{c['doc']}\n\n---\n\n"
    return out


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Document RAG") as demo:
    state = gr.State(None)

    gr.Markdown(
        "# Document RAG\n"
        "Upload any PDF and ask questions — hybrid dense + BM25 retrieval (Recall@10: **91.7%**)"
    )

    with gr.Row():
        # left column — upload & doc management
        with gr.Column(scale=1):
            gr.Markdown("### Upload PDFs")
            file_upload = gr.File(
                label="Select PDFs (max 30)",
                file_types=[".pdf"],
                file_count="multiple",
            )
            upload_btn    = gr.Button("Process PDFs", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)
            chunk_info    = gr.Textbox(
                label="Chunks used",
                value="0 / 5,000 chunks used",
                interactive=False,
            )
            doc_list = gr.Textbox(
                label="Your Documents",
                value="No documents uploaded yet.",
                interactive=False,
                lines=6,
            )
            clear_btn = gr.Button("Clear all documents", variant="stop")

        # right column — query & results
        with gr.Column(scale=2):
            gr.Markdown("### Ask a Question")
            query = gr.Textbox(
                label="Your question",
                placeholder="e.g. What are the payment terms in the lease?",
            )
            doc_filter = gr.Dropdown(
                label="Search in (leave blank to search all documents)",
                choices=[],
                multiselect=True,
                interactive=True,
            )
            ask_btn = gr.Button("Ask", variant="primary")

            answer_box = gr.Textbox(label="Answer", interactive=False, lines=6)
            chunks_box = gr.Markdown(label="Retrieved Chunks")

    # wire up events
    upload_btn.click(
        fn=process_upload,
        inputs=[file_upload, state],
        outputs=[state, upload_status, doc_list, chunk_info, doc_filter],
    )
    ask_btn.click(
        fn=search_and_generate,
        inputs=[query, doc_filter, state],
        outputs=[answer_box, chunks_box, state],
    )
    query.submit(
        fn=search_and_generate,
        inputs=[query, doc_filter, state],
        outputs=[answer_box, chunks_box, state],
    )
    clear_btn.click(
        fn=clear_session,
        inputs=[state],
        outputs=[state, upload_status, doc_list, chunk_info, doc_filter],
    )


if __name__ == "__main__":
    demo.launch()

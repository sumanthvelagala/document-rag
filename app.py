import os
import uuid
from io import BytesIO
import torch
import gradio as gr
import spaces
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM
from cleaners import clean_text
from chunking import sliding_window_chunks

MAX_CHUNKS    = 5000
CHUNK_TOKENS  = 128
CHUNK_OVERLAP = 32
RETRIEVE_K    = 50
RRF_FINAL_K   = 10
LLM_K         = 5
RRF_K         = 60

RAG_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the information "
    "provided in the context below. If the answer is not in the context, say: "
    "'The provided documents don't contain enough information to answer this.'\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


# ── cached globals (loaded once, shared across sessions) ──────────────────────
_embedder  = None
_llm_model = None
_llm_tok   = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-mpnet-base-v2")
    return _embedder

def get_llm():
    global _llm_model, _llm_tok
    if _llm_model is None:
        name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        _llm_tok   = AutoTokenizer.from_pretrained(name)
        _llm_model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    return _llm_model, _llm_tok

@spaces.GPU
def encode(texts):
    return get_embedder().encode(texts, normalize_embeddings=True)

@spaces.GPU
def generate_on_gpu(prompt):
    model, tokenizer = get_llm()
    inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


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
def hybrid_search(query, state):
    if not state["bm25_docs"]:
        return []

    k         = min(RETRIEVE_K, len(state["bm25_docs"]))
    q_emb     = encode(query)
    results   = state["collection"].query(query_embeddings=[q_emb.tolist()], n_results=k)
    dense_ids   = results["ids"][0]
    dense_docs  = results["documents"][0]
    dense_metas = results["metadatas"][0]

    scores   = state["bm25"].get_scores(query.lower().split())
    top_bm25 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    fused = {}
    for rank, (doc_id, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
        if doc_id not in fused:
            fused[doc_id] = {"score": 0.0, "doc": doc, "meta": meta}
        fused[doc_id]["score"] += 2.0 / (RRF_K + rank + 1)

    for rank, idx in enumerate(top_bm25):
        doc_id = state["bm25_ids"][idx]
        if doc_id not in fused:
            fused[doc_id] = {
                "score": 0.0,
                "doc":   state["bm25_docs"][idx],
                "meta":  state["bm25_metas"][idx],
            }
        fused[doc_id]["score"] += 1.0 / (RRF_K + rank + 1)

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:RRF_FINAL_K]


# ── answer generation ─────────────────────────────────────────────────────────
def generate_answer(query, chunks):
    _, tokenizer = get_llm()

    context = ""
    for c in chunks[:LLM_K]:
        context += f"[{c['meta'].get('pdf_id', 'unknown')}]\n{c['doc']}\n\n"

    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "If the answer is not in the context, say so. Do not use outside knowledge."
        )},
        {"role": "user", "content": RAG_PROMPT.format(context=context, question=query)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate_on_gpu(prompt)


# ── gradio event handlers ─────────────────────────────────────────────────────
def process_upload(files, state):
    if state is None:
        state = init_session()
    if not files:
        return state, "No files selected.", _doc_list(state), _chunk_info(state)

    msgs = []
    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            state, msg = ingest_pdf(f.read(), filename, state)
        msgs.append(msg)

    return state, "\n".join(msgs), _doc_list(state), _chunk_info(state)


def search_chunks(query, state):
    if state is None or not state["uploaded"]:
        return "Upload at least one PDF first.", "", state
    if not query.strip():
        return "Enter a question.", "", state

    chunks = hybrid_search(query, state)
    if not chunks:
        return "No results found.", "", state

    return "", _format_chunks(chunks), state


def search_and_generate(query, state):
    if state is None or not state["uploaded"]:
        return "Upload at least one PDF first.", "", state
    if not query.strip():
        return "Enter a question.", "", state

    chunks = hybrid_search(query, state)
    if not chunks:
        return "No results found.", "", state

    answer = generate_answer(query, chunks)
    return answer, _format_chunks(chunks), state


def clear_session(state):
    state = init_session()
    return state, "All documents cleared.", "No documents uploaded yet.", "0 / 5,000 chunks used"


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
            with gr.Row():
                search_btn   = gr.Button("Search chunks", variant="secondary")
                generate_btn = gr.Button("Generate answer", variant="primary")

            answer_box = gr.Textbox(label="Answer", interactive=False, lines=6)
            chunks_box = gr.Markdown(label="Retrieved Chunks")

    # wire up events
    upload_btn.click(
        fn=process_upload,
        inputs=[file_upload, state],
        outputs=[state, upload_status, doc_list, chunk_info],
    )
    search_btn.click(
        fn=search_chunks,
        inputs=[query, state],
        outputs=[answer_box, chunks_box, state],
    )
    generate_btn.click(
        fn=search_and_generate,
        inputs=[query, state],
        outputs=[answer_box, chunks_box, state],
    )
    clear_btn.click(
        fn=clear_session,
        inputs=[state],
        outputs=[state, upload_status, doc_list, chunk_info],
    )


if __name__ == "__main__":
    demo.launch()

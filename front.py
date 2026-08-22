import streamlit as st
import uuid
import numpy as np
from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from cleaners import clean_text
from chunking import sliding_window_chunks

MAX_CHUNKS   = 5000
CHUNK_TOKENS = 128
CHUNK_OVERLAP = 32
RETRIEVE_K   = 50
RRF_FINAL_K  = 10
LLM_K        = 5
RRF_K        = 60


# ── session init ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id  = str(uuid.uuid4())[:8]
    st.session_state.uploaded    = []   # list of filenames
    st.session_state.chunk_count = 0
    st.session_state.bm25_docs   = []
    st.session_state.bm25_metas  = []
    st.session_state.bm25_ids    = []
    st.session_state.bm25        = None


# ── cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-mpnet-base-v2")



# ── in-memory chromadb per session ────────────────────────────────────────────
def get_collection():
    import chromadb
    if "collection" not in st.session_state:
        client = chromadb.Client()
        st.session_state.collection = client.create_collection(
            f"session_{st.session_state.session_id}"
        )
    return st.session_state.collection


# ── pdf ingestion ─────────────────────────────────────────────────────────────
def ingest_pdf(file_bytes, filename):
    if filename in st.session_state.uploaded:
        return 0, "already uploaded"

    raw  = ""
    reader = PdfReader(BytesIO(file_bytes))
    for page in reader.pages:
        raw += (page.extract_text() or "") + "\n"

    text   = clean_text(raw)
    chunks = sliding_window_chunks(text, CHUNK_TOKENS, CHUNK_OVERLAP)

    if st.session_state.chunk_count + len(chunks) > MAX_CHUNKS:
        return 0, f"exceeds {MAX_CHUNKS:,} chunk limit"

    collection = get_collection()
    embedder   = load_embedder()

    for i, chunk in enumerate(chunks):
        emb      = embedder.encode(chunk, normalize_embeddings=True)
        chunk_id = f"{filename}_chunk{i}"
        collection.add(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[emb.tolist()],
            metadatas=[{"pdf_id": filename}],
        )
        st.session_state.bm25_docs.append(chunk)
        st.session_state.bm25_metas.append({"pdf_id": filename})
        st.session_state.bm25_ids.append(chunk_id)

    st.session_state.uploaded.append(filename)
    st.session_state.chunk_count += len(chunks)

    # rebuild bm25 with new docs
    st.session_state.bm25 = BM25Okapi(
        [d.lower().split() for d in st.session_state.bm25_docs]
    )

    return len(chunks), "ok"


# ── hybrid retrieval ──────────────────────────────────────────────────────────
def hybrid_search(query):
    if not st.session_state.bm25_docs:
        return []

    collection = get_collection()
    embedder   = load_embedder()
    k          = min(RETRIEVE_K, len(st.session_state.bm25_docs))

    # dense retrieval
    q_emb   = embedder.encode(query, normalize_embeddings=True)
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=k)
    dense_ids   = results["ids"][0]
    dense_docs  = results["documents"][0]
    dense_metas = results["metadatas"][0]

    # bm25 retrieval
    scores   = st.session_state.bm25.get_scores(query.lower().split())
    top_bm25 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # reciprocal rank fusion (dense weighted 2x)
    fused = {}
    for rank, (doc_id, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
        if doc_id not in fused:
            fused[doc_id] = {"score": 0.0, "doc": doc, "meta": meta}
        fused[doc_id]["score"] += 2.0 / (RRF_K + rank + 1)

    for rank, idx in enumerate(top_bm25):
        doc_id = st.session_state.bm25_ids[idx]
        if doc_id not in fused:
            fused[doc_id] = {
                "score": 0.0,
                "doc":   st.session_state.bm25_docs[idx],
                "meta":  st.session_state.bm25_metas[idx],
            }
        fused[doc_id]["score"] += 1.0 / (RRF_K + rank + 1)

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:RRF_FINAL_K]


# ── answer generation — extractive with chunk-weighted scoring + semantic dedup
def generate_answer(query, chunks):
    embedder  = load_embedder()
    n_chunks  = len(chunks)

    sentences, sources, chunk_weights = [], [], []
    for rank, c in enumerate(chunks):
        w = 1.0 - (rank / n_chunks) * 0.3
        for s in c["doc"].split("."):
            s = s.strip()
            if len(s) > 20:
                sentences.append(s)
                sources.append(c["meta"].get("pdf_id", "unknown"))
                chunk_weights.append(w)

    if not sentences:
        return "Not found in the document."

    q_emb      = embedder.encode(query, normalize_embeddings=True)
    s_embs     = embedder.encode(sentences, normalize_embeddings=True)
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


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Document RAG", page_icon=":material/menu_book:", layout="wide")
st.title(":material/menu_book: Document RAG")
st.caption("Upload your PDFs and ask questions across them — hybrid dense + BM25 retrieval")


# sidebar — upload + session info
with st.sidebar:
    st.header(":material/upload_file: Upload PDFs")

    # chunk usage progress bar
    progress = st.session_state.chunk_count / MAX_CHUNKS
    st.progress(progress, text=f"{st.session_state.chunk_count:,} / {MAX_CHUNKS:,} chunks used")

    uploaded_files = st.file_uploader(
        "Upload PDFs (max 30 PDFs)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            if len(st.session_state.uploaded) >= 30:
                st.warning("30 PDF limit reached.")
                break
            with st.spinner(f"Processing {f.name}..."):
                n, status = ingest_pdf(f.read(), f.name)
            if status == "ok":
                st.success(f":material/check: {f.name} — {n} chunks added")
            elif status == "already uploaded":
                st.info(f"{f.name} already loaded.")
            else:
                st.error(f":material/error: {f.name} — {status}")

    st.divider()
    st.header(":material/folder_open: Your Documents")
    if st.session_state.uploaded:
        for name in st.session_state.uploaded:
            st.markdown(f"- `{name}`")
    else:
        st.caption("No PDFs uploaded yet.")

    st.divider()
    if st.button(":material/delete: Clear all documents", use_container_width=True):
        for key in ["collection", "uploaded", "chunk_count", "bm25_docs",
                    "bm25_metas", "bm25_ids", "bm25"]:
            st.session_state.pop(key, None)
        st.session_state.uploaded    = []
        st.session_state.chunk_count = 0
        st.session_state.bm25_docs   = []
        st.session_state.bm25_metas  = []
        st.session_state.bm25_ids    = []
        st.session_state.bm25        = None
        st.rerun()


# main — query
st.header(":material/search: Ask a Question")
query = st.text_input("Enter your question:", placeholder="e.g. What are the payment terms in the lease?")
ask_clicked = st.button(":material/search: Ask", use_container_width=True, type="primary")

if ask_clicked and query:
    if not st.session_state.uploaded:
        st.warning("Upload at least one PDF first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            chunks = hybrid_search(query)

        if not chunks:
            st.error("No results found.")
        else:
            with st.spinner("Extracting answer..."):
                answer = generate_answer(query, chunks)
            st.subheader(":material/chat: Answer")
            st.write(answer)
            st.divider()

            st.subheader(f":material/list: Top {len(chunks)} Retrieved Chunks")
            for i, result in enumerate(chunks, 1):
                with st.expander(f"#{i} · {result['meta'].get('pdf_id', 'unknown')}"):
                    st.write(result["doc"])

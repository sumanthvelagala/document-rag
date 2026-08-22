import requests
import time
import chromadb
from rank_bm25 import BM25Okapi

API_URL = "http://127.0.0.1:8000/query"
DB_PATH = "/Users/Sumanth/Terminal/DataBase"
TOP_K = 10

# (question, keyword that must appear in retrieved chunk, expected pdf_id)
test_cases = [
    # BERT
    ("How is BERT pre-trained?",                              "masked",            "bert.pdf"),
    ("What tasks does BERT fine-tune on?",                    "fine-tun",          "bert.pdf"),
    ("What is the BERT model architecture?",                  "Transformer",       "bert.pdf"),
    ("What datasets are used to pre-train BERT?",             "BooksCorpus",       "bert.pdf"),

    # Attention Is All You Need
    ("What is the Transformer architecture?",                 "encoder",           "attention_is_all_you_need.pdf"),
    ("What attention mechanism does the Transformer use?",    "multi-head",        "attention_is_all_you_need.pdf"),
    ("What optimizer was used to train the Transformer?",     "Adam",              "attention_is_all_you_need.pdf"),
    ("What is positional encoding in the Transformer?",       "positional",        "attention_is_all_you_need.pdf"),

    # RAG Lewis et al.
    ("How does RAG combine retrieval with generation?",       "retrieval",         "rag_lewis.pdf"),
    ("What retriever does RAG use?",                          "DPR",               "rag_lewis.pdf"),
    ("What datasets are used to evaluate RAG?",               "Natural Questions",  "rag_lewis.pdf"),
    ("What is the generator model used in RAG?",              "BART",              "rag_lewis.pdf"),

    # SciBERT
    ("What corpus was SciBERT trained on?",                   "Semantic Scholar",  "scibert.pdf"),
    ("How does SciBERT compare to BERT on scientific tasks?", "F1",                "scibert.pdf"),
    ("What vocabulary does SciBERT use?",                     "vocabulary",        "scibert.pdf"),
    ("What scientific domains does SciBERT cover?",           "biomedical",        "scibert.pdf"),

    # DPR
    ("What is dense passage retrieval?",                      "passage",           "dpr.pdf"),
    ("How does DPR differ from BM25?",                        "BM25",              "dpr.pdf"),
    ("What encoder architecture does DPR use?",               "BERT",              "dpr.pdf"),
    ("What open domain QA datasets evaluate DPR?",            "Natural Questions",  "dpr.pdf"),

    # ResNet
    ("What problem does residual learning solve in deep networks?", "degradation",  "resnet.pdf"),
    ("How does ResNet use skip connections?",                  "shortcut",          "resnet.pdf"),
    ("What dataset does ResNet evaluate on?",                  "ImageNet",          "resnet.pdf"),
    ("How deep is the deepest ResNet model evaluated?",        "152",               "resnet.pdf"),

    # CLIP
    ("How does CLIP train on image and text pairs?",           "contrastive",       "clip.pdf"),
    ("What dataset is CLIP pre-trained on?",                   "WIT",               "clip.pdf"),
    ("How does CLIP perform zero-shot classification?",        "zero-shot",         "clip.pdf"),
    ("What image encoder does CLIP use?",                      "ViT",               "clip.pdf"),

    # T5
    ("What input and output format does T5 use?",              "text-to-text",      "t5.pdf"),
    ("What pre-training dataset does T5 use?",                 "C4",                "t5.pdf"),
    ("What pre-training objective does T5 use?",               "denoising",         "t5.pdf"),
    ("What is the largest T5 model size evaluated?",           "11",                "t5.pdf"),

    # GPT-3
    ("What is in-context learning in GPT-3?",                  "in-context",        "gpt3.pdf"),
    ("How many parameters does GPT-3 have?",                   "175",               "gpt3.pdf"),
    ("What training data does GPT-3 use?",                     "Common Crawl",      "gpt3.pdf"),
    ("How does GPT-3 perform few-shot learning?",              "few-shot",          "gpt3.pdf"),

    # InstructGPT
    ("How is InstructGPT trained with human feedback?",        "reinforcement",     "instruct_gpt.pdf"),
    ("What is the reward model in InstructGPT?",               "reward",            "instruct_gpt.pdf"),
    ("What labelers do for InstructGPT training?",             "labeler",           "instruct_gpt.pdf"),
    ("How does InstructGPT align language models?",            "alignment",         "instruct_gpt.pdf"),

    # LoRA
    ("How does LoRA reduce the number of trainable parameters?", "low-rank",        "lora.pdf"),
    ("What weights does LoRA keep frozen during training?",    "frozen",            "lora.pdf"),
    ("What is the rank hyperparameter in LoRA?",               "intrinsic",         "lora.pdf"),
    ("What is the inference overhead introduced by LoRA?",     "latency",           "lora.pdf"),

    # ViT
    ("How does ViT split images into tokens?",                 "patch",             "vit.pdf"),
    ("What large dataset does ViT require for training?",      "JFT",               "vit.pdf"),
    ("How does ViT compare to CNNs on ImageNet?",              "convolutional",     "vit.pdf"),
    ("What position encoding does ViT use?",                   "positional",        "vit.pdf"),

    # GAN
    ("How does a GAN train the generator and discriminator?",  "adversarial",       "gan.pdf"),
    ("What are the two networks in a GAN?",                    "discriminator",     "gan.pdf"),
    ("What game-theoretic objective does GAN optimize?",       "minimax",           "gan.pdf"),
    ("What dataset does the original GAN evaluate on?",        "MNIST",             "gan.pdf"),

    # DDPM
    ("How does DDPM generate images through denoising?",       "denoising",         "ddpm.pdf"),
    ("What is the forward diffusion process in DDPM?",         "Markov",            "ddpm.pdf"),
    ("What neural architecture does DDPM use?",                "u-net",             "ddpm.pdf"),
    ("What image datasets does DDPM evaluate on?",             "CIFAR",             "ddpm.pdf"),

    # Word2Vec
    ("What is the skip-gram model in Word2Vec?",               "skip-gram",         "word2vec.pdf"),
    ("How does Word2Vec use negative sampling?",               "negative sampling", "word2vec.pdf"),
    ("What is CBOW in Word2Vec?",                              "CBOW",              "word2vec.pdf"),
    ("What linguistic regularities does Word2Vec capture?",    "king",              "word2vec.pdf"),
]


# ── load all chunks from ChromaDB for BM25 ──────────────────────────────────
def load_corpus():
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_collection("rag_documents")
    data = col.get()
    docs     = data["documents"]
    ids      = data["ids"]
    metas    = data["metadatas"]
    return docs, ids, metas

def build_bm25(docs):
    tokenized = [doc.lower().split() for doc in docs]
    return BM25Okapi(tokenized)

def bm25_query(bm25, docs, metas, question, keyword, pdf_id):
    tokens = question.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    top_docs  = [docs[i]  for i in top_indices]
    top_metas = [metas[i] for i in top_indices]
    return any(
        keyword.lower() in d.lower() and m["pdf_id"] == pdf_id
        for d, m in zip(top_docs, top_metas)
    )


# ── SciBERT retrieval via FastAPI ────────────────────────────────────────────
def scibert_query(question, keyword, pdf_id):
    start = time.time()
    r = requests.post(API_URL, json={"query": question})
    latency = (time.time() - start) * 1000
    results = r.json()["results"]
    hit = any(
        keyword.lower() in c["chunk"].lower() and c["pdf_id"] == pdf_id
        for c in results
    )
    return hit, latency


# ── run eval ─────────────────────────────────────────────────────────────────
def run_eval():
    print("Loading corpus for BM25...")
    docs, ids, metas = load_corpus()
    bm25 = build_bm25(docs)
    print(f"Corpus loaded: {len(docs)} chunks\n")

    scibert_hits, bm25_hits, latencies = 0, 0, []

    print(f"{'#':<4} {'MiniLM':<10} {'BM25':<10} {'Latency':>8}   Question")
    print("-" * 75)

    for i, (question, keyword, pdf_id) in enumerate(test_cases, 1):
        scibert_hit, latency = scibert_query(question, keyword, pdf_id)
        bm25_hit             = bm25_query(bm25, docs, metas, question, keyword, pdf_id)

        scibert_hits += scibert_hit
        bm25_hits    += bm25_hit
        latencies.append(latency)

        s = "PASS" if scibert_hit else "FAIL"
        b = "PASS" if bm25_hit   else "FAIL"
        print(f"{i:<4} {s:<10} {b:<10} {latency:>7.0f}ms   {question}")

    n = len(test_cases)
    print("\n" + "=" * 75)
    print(f"{'Metric':<30} {'MiniLM':>10} {'BM25':>10}")
    print("-" * 50)
    print(f"{'Retrieval accuracy':<30} {scibert_hits/n*100:>9.1f}% {bm25_hits/n*100:>9.1f}%")
    print(f"{'Hits':<30} {scibert_hits:>9}/{n} {bm25_hits:>9}/{n}")
    print(f"{'Avg query latency':<30} {sum(latencies)/len(latencies):>8.0f}ms {'<1ms':>10}")
    print("=" * 75)

    if scibert_hits > bm25_hits:
        print(f"\nMiniLM outperforms BM25 by {scibert_hits - bm25_hits} questions.")
    elif bm25_hits > scibert_hits:
        print(f"\nBM25 outperforms MiniLM by {bm25_hits - scibert_hits} questions.")
    else:
        print("\nBoth methods perform equally.")

if __name__ == "__main__":
    run_eval()

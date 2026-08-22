import requests
import chromadb
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

API_URL    = "http://127.0.0.1:8000/query"
DB_PATH    = "/Users/Sumanth/Terminal/DataBase"
MODEL_PATH = "/Users/Sumanth/work/Mem/models/qwen2.5-1.5b-base"
TOP_K      = 10
MAX_CTX    = 5
MAX_TOKENS = 120

# (question, reference_answer, pdf_id)
test_cases = [
    # BERT
    ("How is BERT pre-trained?",
     "BERT is pre-trained using masked language modeling where random tokens are masked and predicted, and next sentence prediction.",
     "bert.pdf"),
    ("What tasks does BERT fine-tune on?",
     "BERT is fine-tuned on tasks like question answering, natural language inference, and named entity recognition.",
     "bert.pdf"),
    ("What is the BERT model architecture?",
     "BERT uses a multi-layer bidirectional Transformer encoder architecture.",
     "bert.pdf"),
    ("What datasets are used to pre-train BERT?",
     "BERT is pre-trained on BooksCorpus and English Wikipedia.",
     "bert.pdf"),

    # Attention Is All You Need
    ("What is the Transformer architecture?",
     "The Transformer uses an encoder-decoder architecture with stacked self-attention and feed-forward layers.",
     "attention_is_all_you_need.pdf"),
    ("What attention mechanism does the Transformer use?",
     "The Transformer uses multi-head attention applied in parallel with different learned projections.",
     "attention_is_all_you_need.pdf"),
    ("What optimizer was used to train the Transformer?",
     "The Transformer was trained using the Adam optimizer with a custom learning rate schedule.",
     "attention_is_all_you_need.pdf"),
    ("What is positional encoding in the Transformer?",
     "Positional encoding adds position information to token embeddings using sine and cosine functions of different frequencies.",
     "attention_is_all_you_need.pdf"),

    # RAG
    ("How does RAG combine retrieval with generation?",
     "RAG retrieves relevant passages using a dense retriever and conditions a generative model on those passages to produce answers.",
     "rag_lewis.pdf"),
    ("What retriever does RAG use?",
     "RAG uses Dense Passage Retrieval (DPR) as its retrieval component.",
     "rag_lewis.pdf"),
    ("What datasets are used to evaluate RAG?",
     "RAG is evaluated on Natural Questions, WebQuestions, and CuratedTREC open-domain QA datasets.",
     "rag_lewis.pdf"),
    ("What is the generator model used in RAG?",
     "RAG uses BART as its sequence-to-sequence generator model.",
     "rag_lewis.pdf"),

    # SciBERT
    ("What corpus was SciBERT trained on?",
     "SciBERT was trained on a large corpus from Semantic Scholar including computer science and biomedical papers.",
     "scibert.pdf"),
    ("How does SciBERT compare to BERT on scientific tasks?",
     "SciBERT outperforms BERT on scientific NLP tasks achieving higher F1 scores on named entity recognition and relation classification.",
     "scibert.pdf"),
    ("What vocabulary does SciBERT use?",
     "SciBERT uses a domain-specific scientific vocabulary called SCIVOCAB built from scientific text.",
     "scibert.pdf"),
    ("What scientific domains does SciBERT cover?",
     "SciBERT covers biomedical and computer science scientific domains.",
     "scibert.pdf"),

    # DPR
    ("What is dense passage retrieval?",
     "Dense passage retrieval uses dense vector representations from BERT encoders to retrieve relevant passages for open-domain QA.",
     "dpr.pdf"),
    ("How does DPR differ from BM25?",
     "DPR uses learned dense embeddings while BM25 relies on sparse term-frequency matching.",
     "dpr.pdf"),
    ("What encoder architecture does DPR use?",
     "DPR uses BERT as its encoder architecture for both questions and passages.",
     "dpr.pdf"),
    ("What open domain QA datasets evaluate DPR?",
     "DPR is evaluated on Natural Questions, TriviaQA, WebQuestions, and CuratedTREC.",
     "dpr.pdf"),

    # ResNet
    ("What problem does residual learning solve in deep networks?",
     "Residual learning solves the degradation problem where adding more layers makes deep networks harder to optimize.",
     "resnet.pdf"),
    ("How does ResNet use skip connections?",
     "ResNet uses shortcut connections that skip one or more layers and add the input directly to the layer output.",
     "resnet.pdf"),
    ("What dataset does ResNet evaluate on?",
     "ResNet is evaluated on the ImageNet Large Scale Visual Recognition Challenge dataset.",
     "resnet.pdf"),
    ("How deep is the deepest ResNet model evaluated?",
     "The deepest ResNet evaluated is ResNet-152 with 152 layers.",
     "resnet.pdf"),

    # CLIP
    ("How does CLIP train on image and text pairs?",
     "CLIP trains using contrastive learning to maximize cosine similarity between matching image-text pairs.",
     "clip.pdf"),
    ("What dataset is CLIP pre-trained on?",
     "CLIP is pre-trained on WIT, a dataset of 400 million image-text pairs collected from the internet.",
     "clip.pdf"),
    ("How does CLIP perform zero-shot classification?",
     "CLIP performs zero-shot classification by comparing image embeddings to text embeddings of class descriptions.",
     "clip.pdf"),
    ("What image encoder does CLIP use?",
     "CLIP uses a Vision Transformer (ViT) or modified ResNet as its image encoder.",
     "clip.pdf"),

    # T5
    ("What input and output format does T5 use?",
     "T5 uses a text-to-text format where all tasks are framed as converting input text to output text.",
     "t5.pdf"),
    ("What pre-training dataset does T5 use?",
     "T5 is pre-trained on C4, the Colossal Clean Crawled Corpus derived from Common Crawl.",
     "t5.pdf"),
    ("What pre-training objective does T5 use?",
     "T5 uses a span corruption denoising objective where random spans of text are masked and predicted.",
     "t5.pdf"),
    ("What is the largest T5 model size evaluated?",
     "The largest T5 model has 11 billion parameters.",
     "t5.pdf"),

    # GPT-3
    ("What is in-context learning in GPT-3?",
     "In-context learning allows GPT-3 to perform tasks by conditioning on a few examples in the prompt without weight updates.",
     "gpt3.pdf"),
    ("How many parameters does GPT-3 have?",
     "GPT-3 has 175 billion parameters.",
     "gpt3.pdf"),
    ("What training data does GPT-3 use?",
     "GPT-3 is trained on filtered Common Crawl, WebText2, Books1, Books2, and Wikipedia.",
     "gpt3.pdf"),
    ("How does GPT-3 perform few-shot learning?",
     "GPT-3 performs few-shot learning by including a few task examples directly in the input prompt at inference time.",
     "gpt3.pdf"),

    # InstructGPT
    ("How is InstructGPT trained with human feedback?",
     "InstructGPT is trained using reinforcement learning from human feedback with supervised fine-tuning followed by PPO optimization.",
     "instruct_gpt.pdf"),
    ("What is the reward model in InstructGPT?",
     "The reward model is trained on human preference comparisons to score how well responses follow instructions.",
     "instruct_gpt.pdf"),
    ("What labelers do for InstructGPT training?",
     "Human labelers rank and compare model outputs to create preference data for training the reward model.",
     "instruct_gpt.pdf"),
    ("How does InstructGPT align language models?",
     "InstructGPT aligns language models to follow instructions using RLHF to optimize for human preferences.",
     "instruct_gpt.pdf"),

    # LoRA
    ("How does LoRA reduce the number of trainable parameters?",
     "LoRA decomposes weight update matrices into two low-rank matrices, reducing trainable parameters by up to 10000x.",
     "lora.pdf"),
    ("What weights does LoRA keep frozen during training?",
     "LoRA keeps the original pre-trained model weights frozen and only trains the low-rank decomposition matrices.",
     "lora.pdf"),
    ("What is the rank hyperparameter in LoRA?",
     "LoRA uses a rank hyperparameter r based on the hypothesis that weight updates have low intrinsic rank.",
     "lora.pdf"),
    ("What is the inference overhead introduced by LoRA?",
     "LoRA introduces no inference latency because the trained matrices can be merged with the original weights.",
     "lora.pdf"),

    # ViT
    ("How does ViT split images into tokens?",
     "ViT splits images into fixed-size patches, linearly embeds each patch, and processes the sequence with a Transformer.",
     "vit.pdf"),
    ("What large dataset does ViT require for training?",
     "ViT requires large datasets like JFT-300M to train effectively without convolution inductive biases.",
     "vit.pdf"),
    ("How does ViT compare to CNNs on ImageNet?",
     "ViT matches or outperforms convolutional neural networks on ImageNet when pre-trained on sufficiently large datasets.",
     "vit.pdf"),
    ("What position encoding does ViT use?",
     "ViT uses learnable one-dimensional positional embeddings added to patch embeddings.",
     "vit.pdf"),

    # GAN
    ("How does a GAN train the generator and discriminator?",
     "A GAN trains through an adversarial process where the generator tries to fool the discriminator and the discriminator tries to detect fakes.",
     "gan.pdf"),
    ("What are the two networks in a GAN?",
     "A GAN consists of a generator network that creates samples and a discriminator network that classifies real vs fake.",
     "gan.pdf"),
    ("What game-theoretic objective does GAN optimize?",
     "GANs optimize a minimax game where the generator minimizes and the discriminator maximizes the same objective.",
     "gan.pdf"),
    ("What dataset does the original GAN evaluate on?",
     "The original GAN was evaluated on MNIST, TFD, and CIFAR-10 datasets.",
     "gan.pdf"),

    # DDPM
    ("How does DDPM generate images through denoising?",
     "DDPM generates images by iteratively denoising from Gaussian noise through learned reverse diffusion steps.",
     "ddpm.pdf"),
    ("What is the forward diffusion process in DDPM?",
     "The forward diffusion process is a Markov chain that gradually adds Gaussian noise to data over many timesteps.",
     "ddpm.pdf"),
    ("What neural architecture does DDPM use?",
     "DDPM uses a U-Net architecture with attention layers for the denoising network.",
     "ddpm.pdf"),
    ("What image datasets does DDPM evaluate on?",
     "DDPM is evaluated on CIFAR-10 and LSUN datasets.",
     "ddpm.pdf"),

    # Word2Vec
    ("What is the skip-gram model in Word2Vec?",
     "The skip-gram model predicts surrounding context words given a target word.",
     "word2vec.pdf"),
    ("How does Word2Vec use negative sampling?",
     "Word2Vec uses negative sampling to efficiently train by updating only a small number of negative word examples per step.",
     "word2vec.pdf"),
    ("What is CBOW in Word2Vec?",
     "CBOW predicts a target word from its surrounding context words using a bag-of-words representation.",
     "word2vec.pdf"),
    ("What linguistic regularities does Word2Vec capture?",
     "Word2Vec captures linguistic regularities like king minus man plus woman equals queen through vector arithmetic.",
     "word2vec.pdf"),
]


# ── corpus ────────────────────────────────────────────────────────────────────
def load_corpus():
    client = chromadb.PersistentClient(path=DB_PATH)
    col    = client.get_collection("scientific_paper_RAG")
    data   = col.get()
    return data["documents"], data["ids"], data["metadatas"]

def build_bm25(docs):
    return BM25Okapi([doc.lower().split() for doc in docs])

def bm25_top_chunks(bm25, docs, metas, question):
    scores  = bm25.get_scores(question.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    return [{"chunk": docs[i], "pdf_id": metas[i]["pdf_id"]} for i in top_idx]

def hybrid_top_chunks(question):
    return requests.post(API_URL, json={"query": question}).json()["results"]


# ── Qwen ──────────────────────────────────────────────────────────────────────
def load_qwen():
    print(f"Loading Qwen from {MODEL_PATH} ...")
    device    = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    model.eval()
    print(f"Qwen loaded on {device}\n")
    return tokenizer, model, device

def generate_answer(tokenizer, model, device, chunks, question):
    context = "\n\n".join(c["chunk"] for c in chunks[:MAX_CTX])
    prompt = (
        f"Read the context below and answer the question using ONLY information from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer (use only the context above):"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # truncate at first sentence end so base model doesn't ramble
    for sep in ["\n", ". ", "! ", "? "]:
        if sep in raw:
            raw = raw.split(sep)[0] + ("." if sep == ". " else "")
            break
    return raw


# ── cosine similarity ─────────────────────────────────────────────────────────
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── eval ──────────────────────────────────────────────────────────────────────
def run_eval(test_cases=test_cases):
    print("Loading corpus for BM25...")
    docs, ids, metas = load_corpus()
    bm25 = build_bm25(docs)
    print(f"Corpus: {len(docs)} chunks\n")

    tokenizer, model, device = load_qwen()

    print("Loading sentence embedder for answer scoring...")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    print("Ready.\n")

    n = len(test_cases)
    bm25_sims, hybrid_sims = [], []

    print(f"{'#':<4} {'Hybrid':>8} {'BM25':>8}   Question")
    print("-" * 75)

    for i, (question, reference, pdf_id) in enumerate(test_cases, 1):
        bm25_chunks   = bm25_top_chunks(bm25, docs, metas, question)
        hybrid_chunks = hybrid_top_chunks(question)

        bm25_answer   = generate_answer(tokenizer, model, device, bm25_chunks,   question)
        hybrid_answer = generate_answer(tokenizer, model, device, hybrid_chunks, question)

        ref_emb    = embedder.encode(reference,     normalize_embeddings=True)
        bm25_emb   = embedder.encode(bm25_answer,   normalize_embeddings=True)
        hybrid_emb = embedder.encode(hybrid_answer, normalize_embeddings=True)

        bm25_sim   = cosine_sim(ref_emb, bm25_emb)
        hybrid_sim = cosine_sim(ref_emb, hybrid_emb)

        bm25_sims.append(bm25_sim)
        hybrid_sims.append(hybrid_sim)

        print(f"{i:<4} {hybrid_sim:>7.3f}  {bm25_sim:>7.3f}   {question}")

    avg_hybrid = sum(hybrid_sims) / n
    avg_bm25   = sum(bm25_sims)   / n

    print("\n" + "=" * 75)
    print(f"{'Metric':<35} {'Hybrid+LLM':>12} {'BM25+LLM':>10}")
    print("-" * 60)
    print(f"{'Avg cosine similarity':<35} {avg_hybrid:>12.3f} {avg_bm25:>10.3f}")
    print(f"{'Questions where hybrid wins':<35} {sum(h > b for h, b in zip(hybrid_sims, bm25_sims)):>12} / {n}")
    print(f"{'Questions where BM25 wins':<35} {sum(b > h for h, b in zip(hybrid_sims, bm25_sims)):>12} / {n}")
    print("=" * 75)

    delta = avg_hybrid - avg_bm25
    print(f"\nHybrid RRF + LLM scores {delta:+.3f} cosine similarity vs BM25 + LLM.")


if __name__ == "__main__":
    import argparse, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly sample N questions (0 = run all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sample > 0:
        random.seed(args.seed)
        sampled = random.sample(test_cases, min(args.sample, len(test_cases)))
        run_eval(sampled)
    else:
        run_eval(test_cases)

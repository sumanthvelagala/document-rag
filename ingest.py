import requests
from pathlib import Path
from pypdf import PdfReader
from cleaners import clean_text
from chunking import sliding_window_chunks

API_URL    = "http://127.0.0.1:8000/store"
PAPERS_DIR = Path(__file__).parent / "papers"

def extract_chunks(pdf_path: Path, token_limit=128, overlap=32):
    reader = PdfReader(str(pdf_path))

    # extract and clean all text
    raw = ""
    for page in reader.pages:
        raw += (page.extract_text() or "") + "\n"
    text = clean_text(raw)

    # universal sliding window — no domain-specific title detection
    chunks = sliding_window_chunks(text, token_limit, overlap)

    # return as (title, chunk) tuples — title is filename for traceability
    return [("General", c) for c in chunks]

def ingest(pdf_path: Path):
    print(f"\nProcessing: {pdf_path.name}")
    chunks = extract_chunks(pdf_path)
    print(f"  {len(chunks)} chunks to store")

    payload = {
        "pdf_id": pdf_path.name,
        "chunks": [{"title": t, "chunk": c} for t, c in chunks]
    }

    r = requests.post(API_URL, json=payload)
    if r.status_code == 200:
        print(f"  Stored in ChromaDB")
    else:
        print(f"  Failed: {r.status_code} {r.text}")

if __name__ == "__main__":
    import argparse, chromadb

    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="Wipe ChromaDB collection before ingesting")
    args = parser.parse_args()

    if args.fresh:
        client = chromadb.PersistentClient(path="/Users/Sumanth/Terminal/DataBase")
        try:
            client.delete_collection("rag_documents")
            print("Collection wiped.\n")
        except Exception:
            print("Collection not found, starting fresh.\n")

    pdfs = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} papers to ingest")
    for pdf in sorted(pdfs):
        ingest(pdf)
    print("\nAll papers ingested. Run: python eval.py")

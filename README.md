# RAG for Scientific Research

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer scientific questions based on uploaded PDF documents. The system semantically chunks the PDF content and stores it in a vector database. When a query is submitted, it performs a semantic similarity search to retrieve the most relevant chunks. These retrieved chunks are then passed to a Large Language Model (LLM), which generates a concise and informative answer grounded in the retrieved context.
- Accepts multiple PDF's

When a PDF is uploaded, the system first processes the file by extracting its content and semantically chunking it into meaningful sections using pre-defined cleaning and chunking logic. These chunks are then embedded into vector representations using a tokenizer and stored in a Vector Database (Chroma/FAISS) for efficient similarity search.

Once stored, users can input a custom scientific question. The system uses semantic similarity to compare the query with all stored document embeddings and retrieves the top-k most relevant chunks. These retrieved chunks are compiled as context and sent to a Large Language Model (LLM) (via Hugging Face pipeline) along with the original question.

The LLM generates an answer that is both grounded in the retrieved context and enriched with its own background knowledge. The final answer explicitly highlights which PDF(s) the information came from, making the response traceable and reliable.

RAG model used for tokenization : allenai/scibert_scivocab_uncased
LLM Used : TinyLlama/TinyLlama-1.1B-Chat-v1.0
Database : ChromaDB
API for DB: FastAPI
FrontEnd: Streamlit

-- pdf is read using unstructured.partition.pdf

# Features
Upload one or more scientific PDFs
Semantic chunking based on content structure
Cleaning logic to remove irrelevant headers, footers, or citations
Chunk embedding and storage using a vector database
semantic search over stored chunks
Query answering using LLM with contextual awareness
Chunk preview and section-based filtering for better interpretability



# Steps to download and use the project
1.Create Virtual Environment and activate it
python -m venv venv
source venv/bin/activate

2.Install Requirements
pip install -r requirements.txt

3.Start API for DB
uvicorn main:app --reload

4.Run Streamlit Front End
Streamlit run front.py



-change the path to DataBase while using
-Change the Address of Database in front.py
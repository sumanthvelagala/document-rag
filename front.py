import streamlit as st
import requests
from cleaners import is_header_or_footer,clean_between_title_and_abstract,clean_citations
from chunking import chunk_elements
from unstructured.partition.pdf import partition_pdf
from io import BytesIO
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


st.title("RAG for Scientific Research")
st.header("Upload Scientific PDF")
pdf_files = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)

if pdf_files:
    for pdf_file in pdf_files:
        pdf_bytes = pdf_file.read()
        pdf_id = pdf_file.name
        with st.spinner("Extracting content using"):
            elements = partition_pdf(file=BytesIO(pdf_bytes), strategy="auto")
            cleaned_elements = [el for el in elements if not is_header_or_footer(el)]
            cleaned_elements = clean_between_title_and_abstract(cleaned_elements)
            cleaned_elements = clean_citations(cleaned_elements)
            chuncked_data = chunk_elements(cleaned_elements)
            st.success(":material/check: PDF parsed and chunked!")
            #st.text_area("First Chunk Preview", chuncked_data[0][1][:1000], height=200)
        
            with st.spinner(f"Sending {pdf_id} PDF chunks to backend..."):
                payload = {
                        "pdf_id":pdf_id,
                        "chunks":[{"title": title, "chunk": chunk} for title, chunk in chuncked_data]}
                response = requests.post("http://127.0.0.1:8000/store", json=payload)
                if response.status_code == 200:
                    st.success(f"Chunks for {pdf_id} embedded and stored in DB.")
                else:
                    st.error(f"Failed to store chunks fro {pdf_id}in DB.")
            
        # Filter Chunks by Section Title
        st.subheader(":material/search: Filter Chunks by Section Title")
        titles = list(set(title for title, _ in chuncked_data))  # Unique titles
        title_to_search = st.selectbox("Select a section title:", sorted(titles))

        matching_chunks = [chunk for title, chunk in chuncked_data if title == title_to_search]

        if matching_chunks:
            st.success(f"Found {len(matching_chunks)} chunk(s) for '{title_to_search}':")
            for i, chunk in enumerate(matching_chunks, 1):
                st.markdown(f"### 🔹 Chunk {i}")
                st.write(chunk)
        else:
            st.warning("No chunks found for the selected title.")

# Step 2: Query
st.header(":material/search: Ask a Question")
query = st.text_input("Enter your query:")
if "context" not in st.session_state:
    st.session_state.context = ""
if st.button("Search") and query:
    with st.spinner("Embedding query and retrieving chunks..."):
        response = requests.post("http://127.0.0.1:8000/query", json={"query": query})
        if response.status_code == 200:
            results = response.json()["results"]
            st.success("Top Relevant Chunks:")
            st.session_state.context = ""
            for i, result in enumerate(results, 1):
                st.markdown(f"### 🔹 {i}.")
                st.write(result['chunk'])
                st.write(f"📄 PDF ID: {result['pdf_id']}")

              
                st.session_state.context += (
                f"[PDF ID: {result['pdf_id']}]\n"
                f"[Section Title: {result['title']}]\n"
                f"{result['chunk']}\n\n"
)
        else:
            st.error("Failed to get results from API.")




@st.cache_resource
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return pipeline("text-generation", tokenizer=tokenizer, model=model),tokenizer

llm_pipeline ,tokenizer = load_llm()


if st.button(":material/chat: Generate Answer with Hugging Face LLM"):
    messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful scientific assistant. Use only the provided context "
            "to answer the user's question. Context includes chunks from multiple PDFs, "
            "each marked with [PDF ID: ...] and [Section Title: ...].\n\n"
            "You may also include your own background knowledge **if helpful**, but make sure to clearly separate it. "
            "Always cite the [PDF ID] when referring to specific content."
        )
    },
    {
        "role": "user",
        "content": (
            f"Context:\n{st.session_state.context}\n\n"
            f"Question: {query}\n\n"
            "Please respond with a well-supported answer using the context above. Clearly indicate which PDF(s) the answer is based on."
        )
    }
]


    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    st.markdown(" :material/code: Prompt Sent to LLM:")
    st.code(prompt, language="markdown")

    with st.spinner("Generating answer..."):
        result = llm_pipeline(prompt, max_new_tokens=600, do_sample=False)
        answer = result[0]['generated_text'].split("<|assistant|>")[-1].strip()

        st.markdown(":material/chat: Answer:")
        st.write(answer)

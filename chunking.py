"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
#semantic chunking
def count_token(text):
    return len(tokenizer.encode(text,truncation= False))

def chunk_elements(elements,token_limit=512):
    chunks = []
    current_chunk = " "
    current_title = "Introduction"

    for i in elements:
        if i.category != "Title":
            temp_chunk = current_chunk+" "+i.text
            if count_token(temp_chunk) > token_limit:
                chunks.append((current_title,current_chunk))
                current_chunk = i.text
            else:
                current_chunk = temp_chunk
                
        elif i.category == "Title":
            if current_chunk:
                chunks.append((current_title,current_chunk))
            current_title = i.text
            current_chunk = ""
    if current_chunk.strip():
        chunks.append((current_title, current_chunk.strip()))
    return chunks


"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def count_tokens(text):
    return len(tokenizer.encode(text, truncation=False))

def chunk_elements(elements, token_limit=200):
    chunks = []
    current_chunk = ""
    current_title = "General"

    for el in elements:
        if el.category == "Title":
            if current_chunk.strip():
                chunks.append((current_title, current_chunk.strip()))
                current_chunk = ""
            current_title = el.text.strip() or current_title
            continue

        if el.category in ("NarrativeText", "Paragraph"):
            paragraph = el.text.strip()
            if not paragraph:
                continue

            temp = f"{current_chunk} {paragraph}".strip() if current_chunk else paragraph

            if count_tokens(temp) > token_limit:
                chunks.append((current_title, current_chunk.strip()))
                current_chunk = paragraph
            else:
                current_chunk = temp

    if current_chunk.strip():
        chunks.append((current_title, current_chunk.strip()))

    return chunks





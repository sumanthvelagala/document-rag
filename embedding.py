from transformers import AutoTokenizer, AutoModel
import torch


tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

model.eval()

def tokenize(text):

    inputs = tokenizer(text, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
    return cls_embedding.squeeze().numpy()
    
def embeddings(chunks):
    embedded_chunks = []
    for title, chunk in chunks:
        embedding = tokenize(chunk)
        embedded_chunks.append((title, chunk, embedding))
    return embedded_chunks

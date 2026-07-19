# day61_hf_basics/inference.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

print("[HUGGINGFACE TOKENIZER + MODEL INFERENCE]\n")

# Load tokenizer and model
model_name = "distilbert-base-uncased"
print(f"Loading: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModel.from_pretrained(model_name)
model.eval()

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Tokenize text
text   = "Machine learning transforms raw data into intelligent predictions"
tokens = tokenizer(text, return_tensors="pt")

print(f"[TOKENIZATION]")
print(f"Input text:    {text}")
print(f"Token IDs:     {tokens['input_ids'][0].tolist()}")
print(f"Tokens:        {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
print(f"Token count:   {len(tokens['input_ids'][0])}\n")

# Get embeddings
with torch.no_grad():
    outputs = model(**tokens)

embeddings = outputs.last_hidden_state
cls_embedding = embeddings[0][0].numpy()  # CLS token

print(f"[EMBEDDINGS]")
print(f"Output shape:  {embeddings.shape}  (batch x tokens x hidden_size)")
print(f"CLS embedding: {cls_embedding[:5].round(4)}... (first 5 of {len(cls_embedding)} dims)")
print(f"Embedding norm: {np.linalg.norm(cls_embedding):.4f}\n")

# Cosine similarity between two sentences
def get_embedding(text: str) -> np.ndarray:
    tokens  = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[0][0].numpy()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sentences = [
    ("I love machine learning", "I enjoy deep learning"),
    ("I love machine learning", "The weather is nice today"),
    ("Python is great for AI", "Artificial intelligence uses Python"),
]

print("[SEMANTIC SIMILARITY]")
for s1, s2 in sentences:
    e1  = get_embedding(s1)
    e2  = get_embedding(s2)
    sim = cosine_sim(e1, e2)
    print(f"'{s1[:30]}' vs '{s2[:30]}'")
    print(f"Similarity: {sim:.4f}\n")
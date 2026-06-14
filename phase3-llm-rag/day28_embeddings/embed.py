import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> np.ndarray:
    return model.encode(text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    sentences = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "Machine learning is fascinating",
        "The dog ran through the park",
    ]

    print("[EMBEDDING SIMILARITY]\n")
    reference = get_embedding(sentences[0])
    print(f"Reference: '{sentences[0]}'\n")
    print(f"Embedding dimensions: {len(reference)}\n")

    for sent in sentences[1:]:
        emb = get_embedding(sent)
        sim = cosine_similarity(reference, emb)
        bar = "█" * int(sim * 20)
        print(f"Similarity: {sim:.4f} {bar}")
        print(f"Text:       '{sent}'\n")
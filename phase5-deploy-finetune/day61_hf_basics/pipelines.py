# day61_hf_basics/pipelines.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import pipeline
import time

print("[HUGGINGFACE PIPELINES]\n")

# 1. Sentiment Analysis
print("1. SENTIMENT ANALYSIS")
print("-" * 40)
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

reviews = [
    "This product is absolutely amazing!",
    "Terrible quality, complete waste of money.",
    "It's okay, nothing special.",
]

for review in reviews:
    start  = time.time()
    result = sentiment(review)[0]
    elapsed = time.time() - start
    print(f"Text:  {review[:50]}")
    print(f"Label: {result['label']} | Score: {result['score']:.4f} | Time: {elapsed:.2f}s\n")


# 2. Text Generation
print("\n2. TEXT GENERATION")
print("-" * 40)
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=30,
    do_sample=False
)

prompts = [
    "Machine learning is",
    "The best AI applications",
]

for prompt in prompts:
    result = generator(prompt)[0]["generated_text"]
    print(f"Prompt: {prompt}")
    print(f"Output: {result}\n")


# 3. Zero-shot classification
print("\n3. ZERO-SHOT CLASSIFICATION")
print("-" * 40)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text   = "The new iPhone has an amazing camera and battery life"
labels = ["technology", "sports", "food", "politics"]
result = classifier(text, candidate_labels=labels)

print(f"Text: {text}")
print("Scores:")
for label, score in zip(result["labels"], result["scores"]):
    bar = "#" * int(score * 20)
    print(f"  {label:12s}: {score:.4f} {bar}")
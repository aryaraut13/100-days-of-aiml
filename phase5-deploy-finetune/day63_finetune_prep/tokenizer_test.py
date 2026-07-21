import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import AutoTokenizer
import json

print("[TOKENIZER ANALYSIS FOR FINE-TUNING]\n")

model_name = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)

print(f"Model:      {model_name}")
print(f"Vocab size: {tokenizer.vocab_size:,}")
print(f"Max length: {tokenizer.model_max_length}\n")

# Load dataset
with open("ecommerce_support_dataset.json") as f:
    samples = [json.loads(line) for line in f]

texts = [s["text"] for s in samples]

# Tokenize all texts
print("[TOKENIZATION ANALYSIS]")
token_lengths = []
for text in texts:
    tokens = tokenizer(text, truncation=True, max_length=128)
    token_lengths.append(len(tokens["input_ids"]))

print(f"Avg token length: {sum(token_lengths)/len(token_lengths):.1f}")
print(f"Max token length: {max(token_lengths)}")
print(f"Min token length: {min(token_lengths)}")
print(f"Recommended max_length: {max(token_lengths) + 10} (add 10 for safety)\n")

# Show tokenization of sample texts
print("[SAMPLE TOKENIZATION]")
for text in texts[:3]:
    tokens = tokenizer(text)
    words  = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
    print(f"Text:   {text}")
    print(f"Tokens: {words}")
    print(f"Count:  {len(words)}\n")

# Batch tokenization (production approach)
print("[BATCH TOKENIZATION]")
batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)
print(f"Batch input_ids shape: {batch['input_ids'].shape}")
print(f"Padding token ID:      {tokenizer.pad_token_id}")
print(f"All sequences padded to length: {batch['input_ids'].shape[1]}")
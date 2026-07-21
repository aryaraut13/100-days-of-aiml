import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from datasets import Dataset
import pandas as pd

print("[DATASET PREPARATION FOR FINE-TUNING]\n")

# Create ecommerce support dataset
raw_data = [
    {"text": "How do I return a product?",
     "label": 0, "category": "returns"},
    {"text": "I want to send back my order",
     "label": 0, "category": "returns"},
    {"text": "Can I get a refund on my purchase?",
     "label": 0, "category": "returns"},
    {"text": "The item I received is damaged",
     "label": 0, "category": "returns"},
    {"text": "When will my order arrive?",
     "label": 1, "category": "shipping"},
    {"text": "Track my shipment please",
     "label": 1, "category": "shipping"},
    {"text": "My package is delayed",
     "label": 1, "category": "shipping"},
    {"text": "What delivery options do you have?",
     "label": 1, "category": "shipping"},
    {"text": "What payment methods do you accept?",
     "label": 2, "category": "payment"},
    {"text": "Can I pay with UPI?",
     "label": 2, "category": "payment"},
    {"text": "My payment failed at checkout",
     "label": 2, "category": "payment"},
    {"text": "Do you accept credit cards?",
     "label": 2, "category": "payment"},
    {"text": "Is my product under warranty?",
     "label": 3, "category": "warranty"},
    {"text": "How do I claim warranty?",
     "label": 3, "category": "warranty"},
    {"text": "My product stopped working after 2 months",
     "label": 3, "category": "warranty"},
    {"text": "What is the warranty period?",
     "label": 3, "category": "warranty"},
]

label_names = ["returns", "shipping", "payment", "warranty"]

# Convert to HuggingFace Dataset
df      = pd.DataFrame(raw_data)
dataset = Dataset.from_pandas(df)

print(f"[DATASET INFO]")
print(f"Total samples:  {len(dataset)}")
print(f"Features:       {list(dataset.features.keys())}")
print(f"Label classes:  {label_names}")

# Show distribution
print(f"\n[CLASS DISTRIBUTION]")
for i, name in enumerate(label_names):
    count = sum(1 for x in raw_data if x["label"] == i)
    bar   = "#" * count
    print(f"  {name:10s} (label {i}): {count} samples {bar}")

# Train/test split
split   = dataset.train_test_split(test_size=0.25, seed=42)
train   = split["train"]
test    = split["test"]

print(f"\n[SPLIT]")
print(f"Train: {len(train)} samples")
print(f"Test:  {len(test)} samples")

# Save dataset
dataset.to_json("ecommerce_support_dataset.json")
print(f"\nDataset saved -> ecommerce_support_dataset.json")

# Show samples
print(f"\n[SAMPLE DATA]")
for i in range(3):
    sample = train[i]
    print(f"  Text:     {sample['text']}")
    print(f"  Label:    {sample['label']} ({label_names[sample['label']]})")
    print()
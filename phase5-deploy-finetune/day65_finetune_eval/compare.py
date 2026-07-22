# day65_finetune_eval/compare.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

print("[FINE-TUNED MODEL EVALUATION]\n")

model_name  = "distilbert-base-uncased"
label_names = ["returns", "shipping", "payment", "warranty"]
tokenizer   = AutoTokenizer.from_pretrained(model_name)

# Load base model (before fine-tuning)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=4
)
base_model.eval()

# Load fine-tuned model (after fine-tuning)
ft_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=4
)
ft_model = PeftModel.from_pretrained(ft_model, "../day64_finetune_lora/lora_adapter")
ft_model.eval()


def predict(model, text: str) -> tuple:
    tokens  = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**tokens).logits
    probs  = torch.softmax(logits, dim=-1)[0].numpy()
    pred   = np.argmax(probs)
    return label_names[pred], float(probs[pred])


# Test queries
test_queries = [
    ("I want to send back my damaged item",      "returns"),
    ("My shipment is stuck in transit",           "shipping"),
    ("Transaction declined at checkout",          "payment"),
    ("Product broke after 1 month of use",        "warranty"),
    ("Can I get my money back?",                  "returns"),
    ("Where is my package?",                      "shipping"),
]

print(f"{'Query':45s} {'Expected':10s} {'Base':10s} {'Fine-tuned':12s}")
print("-" * 85)

base_correct = 0
ft_correct   = 0

for query, expected in test_queries:
    base_pred, base_conf = predict(base_model, query)
    ft_pred,   ft_conf   = predict(ft_model,   query)

    base_ok = "OK" if base_pred == expected else "WRONG"
    ft_ok   = "OK" if ft_pred   == expected else "WRONG"

    if base_pred == expected: base_correct += 1
    if ft_pred   == expected: ft_correct   += 1

    print(f"{query[:44]:45s} {expected:10s} {base_ok:10s} {ft_ok:12s}")

print("-" * 85)
print(f"\n[RESULTS]")
print(f"Base model accuracy:        {base_correct}/{len(test_queries)} ({base_correct/len(test_queries):.0%})")
print(f"Fine-tuned model accuracy:  {ft_correct}/{len(test_queries)}  ({ft_correct/len(test_queries):.0%})")
print(f"Improvement:                +{(ft_correct-base_correct)/len(test_queries):.0%}")
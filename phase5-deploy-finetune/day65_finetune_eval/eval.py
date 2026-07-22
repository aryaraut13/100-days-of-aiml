# day65_finetune_eval/eval.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

model_name  = "distilbert-base-uncased"
label_names = ["returns", "shipping", "payment", "warranty"]
tokenizer   = AutoTokenizer.from_pretrained(model_name)

ft_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
ft_model = PeftModel.from_pretrained(ft_model, "../day64_finetune_lora/lora_adapter")
ft_model.eval()


def predict_with_confidence(text: str) -> dict:
    tokens  = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = ft_model(**tokens).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    pred  = np.argmax(probs)
    return {
        "text":       text,
        "prediction": label_names[pred],
        "confidence": float(probs[pred]),
        "all_scores": {label_names[i]: float(probs[i]) for i in range(4)}
    }


if __name__ == "__main__":
    print("[FINE-TUNED MODEL — CONFIDENCE ANALYSIS]\n")

    queries = [
        "I need to return this broken item",
        "My delivery is late by 3 days",
        "Can I pay using net banking?",
        "The product stopped working after 2 weeks",
        "I accidentally ordered the wrong size",  # edge case
    ]

    for query in queries:
        result = predict_with_confidence(query)
        print(f"Query:      {result['text']}")
        print(f"Prediction: {result['prediction']} ({result['confidence']:.1%} confident)")
        print(f"All scores: {', '.join(f'{k}:{v:.2f}' for k,v in result['all_scores'].items())}")
        print()
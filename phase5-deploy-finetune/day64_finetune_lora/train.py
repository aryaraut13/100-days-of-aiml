import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

print("[LORA FINE-TUNING — ECOMMERCE INTENT CLASSIFIER]\n")

# Load dataset
with open("../day63_finetune_prep/ecommerce_support_dataset.json") as f:
    samples = [json.loads(line) for line in f]

label_names = ["returns", "shipping", "payment", "warranty"]
model_name  = "distilbert-base-uncased"

# Prepare dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=64
    )

dataset = Dataset.from_list(samples)
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text", "category"])
dataset = dataset.rename_column("label", "labels")
dataset = dataset.with_format("torch")
split   = dataset.train_test_split(test_size=0.25, seed=42)

# Load model with LoRA
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=4
)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"], bias="none"
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

# Training arguments
args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=5,
    report_to="none",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


# Train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

print("\n[TRAINING]\n")
trainer.train()

print("\n[EVALUATION]")
results = trainer.evaluate()
print(f"Final accuracy: {results['eval_accuracy']:.4f}")

# Save the model
model.save_pretrained("./lora_adapter")
print("\nLoRA adapter saved -> ./lora_adapter")
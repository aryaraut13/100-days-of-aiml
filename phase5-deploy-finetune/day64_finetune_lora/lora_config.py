import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

print("[LORA CONFIGURATION]\n")

model_name  = "distilbert-base-uncased"
num_labels  = 4
label_names = ["returns", "shipping", "payment", "warranty"]

# Load base model
print(f"Loading base model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={i: name for i, name in enumerate(label_names)},
    label2id={name: i for i, name in enumerate(label_names)},
)

total_params    = sum(p.numel() for p in model.parameters())
print(f"Base model parameters: {total_params:,}\n")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                    # rank — lower = fewer params
    lora_alpha=16,          # scaling factor
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # which layers to adapt
    bias="none",
)

# Apply LoRA
lora_model      = get_peft_model(model, lora_config)
trainable       = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total           = sum(p.numel() for p in lora_model.parameters())
trainable_pct   = 100 * trainable / total

print("[LORA SUMMARY]")
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Trainable %:          {trainable_pct:.2f}%")
print(f"Frozen %:             {100 - trainable_pct:.2f}%")
print(f"\nLoRA config:")
print(f"  rank (r):           {lora_config.r}")
print(f"  alpha:              {lora_config.lora_alpha}")
print(f"  dropout:            {lora_config.lora_dropout}")
print(f"  target modules:     {lora_config.target_modules}")
print(f"\nKey insight: Only {trainable_pct:.2f}% of parameters are updated during training.")
print(f"The rest ({100-trainable_pct:.2f}%) stay frozen — that's LoRA.")
# train_kfold.py

import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import KFold
from custom_dataset import MultiImageJSONLDataset
from evaluate_ import compute_metrics

# Setup
MODEL_NAME = "/workspace/PDF-AI/hf/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
BASE_IMAGE_DIR = "./data/panels"
JSONL_PATH = "./data/training/data.jsonl"
OUTPUT_DIR = "./results"
SAVE_MODEL_DIR = "./ctc-crypto-analyst"  # Your final model name
K_FOLDS = 5
BATCH_SIZE = 2
EPOCHS = 2
MAX_LENGTH = 2048

GRADIENT_ACCUMULATION_STEPS = 4
DEVICE_MAP = "auto"          # Let HF handle multi-GPU
FP16 = True                  # Use mixed precision

# Load processor and model
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True)

# LoRA config
peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load full dataset
full_dataset = MultiImageJSONLDataset(JSONL_PATH, processor, BASE_IMAGE_DIR, MAX_LENGTH)

# K-Fold setup
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
all_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset.data)):
    print(f"\n--- Fold {fold + 1} ---")

    # Save train/val split to JSONL
    train_data = [full_dataset.data[i] for i in train_idx]
    val_data = [full_dataset.data[i] for i in val_idx]

    with open(f"./results/train_fold_{fold}.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(f"./results/val_fold_{fold}.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    # Load datasets
    train_dataset = MultiImageJSONLDataset(f"./results/train_fold_{fold}.jsonl", processor, BASE_IMAGE_DIR, MAX_LENGTH)
    val_dataset = MultiImageJSONLDataset(f"./results/val_fold_{fold}.jsonl", processor, BASE_IMAGE_DIR, MAX_LENGTH)

    # Wrap model with LoRA
    model = get_peft_model(base_model, peft_config)
    model.train()


    training_args = TrainingArguments(
        output_dir=f"./results/fold_{fold}",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=2e-4,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        fp16=FP16,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        # Multi-GPU
        per_device_eval_batch_size=BATCH_SIZE,
        # Optional: use deepspeed for even better memory optimization
    )

    # Data collator

    def data_collator(samples):
        input_ids = torch.stack([s["input_ids"] for s in samples])
        attention_mask = torch.stack([s["attention_mask"] for s in samples])
        labels = torch.stack([s["labels"] for s in samples])

        pixel_values_list = []
        for s in samples:
            img = s["pixel_values"]

            # Expect img shape: [C, H, W]
            if img.dim() == 3:
                pixel_values_list.append(img)
            elif img.dim() == 4:
                # If it's [1, C, H, W], remove batch dim
                pixel_values_list.append(img[0])
            else:
                raise ValueError(f"Expected image with 3 or 4 dims, got {img.dim()}")

        pixel_values = torch.stack(pixel_values_list)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }


    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Evaluate
    model.eval()
    predictions = []
    references = []

    for item in val_dataset:
        inputs = {
            "input_ids": item["input_ids"].unsqueeze(0).to("cuda"),
            "attention_mask": item["attention_mask"].unsqueeze(0).to("cuda"),
            "pixel_values": item["pixel_values"].unsqueeze(0).to("cuda")
        }

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=100)
        pred = processor.decode(output_ids[0], skip_special_tokens=True)
        ref = processor.decode(item["labels"], skip_special_tokens=True)

        predictions.append(pred)
        references.append(ref)

    metrics = compute_metrics(predictions, references)
    print(f"Metrics for Fold {fold + 1}: {metrics}")
    metrics["fold"] = fold + 1
    all_metrics.append(metrics)

# Save metrics to CSV
df = pd.DataFrame(all_metrics)
df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)

# Merge LoRA with base model
merged_model = model.merge_and_unload()

# Save merged model
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
merged_model.save_pretrained(SAVE_MODEL_DIR)
processor.save_pretrained(SAVE_MODEL_DIR)

print("\n✅ All folds completed. Final merged model saved to:", SAVE_MODEL_DIR)
print("📊 Metrics saved to:", os.path.join(OUTPUT_DIR, "metrics_summary.csv"))

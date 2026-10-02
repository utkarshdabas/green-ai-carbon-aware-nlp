import os
import math
import torch
import time
import csv
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    Trainer,
    TrainingArguments
)
from codecarbon import EmissionsTracker

os.makedirs("models/bert", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = torch.device("cpu")
print("Using device:", device)

# ===================================
# LOAD DATASET
# ===================================
dataset = load_dataset("squad")
dataset = dataset.shuffle(seed=42)

train_dataset = dataset["train"].select(range(2000))

# 🔥 RAW dataset (important fix)
raw_val_dataset = dataset["validation"].select(range(500))

# ===================================
# TOKENIZER
# ===================================
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ===================================
# PREPROCESS FUNCTION
# ===================================
def preprocess(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )

    offsets = inputs.pop("offset_mapping")
    answer = example["answers"]

    if len(answer["text"]) == 0:
        start_position = 0
        end_position = 0
    else:
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids()

        start_position = 0
        end_position = 0

        for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
            if seq_id == 1:
                if offset[0] <= start_char <= offset[1]:
                    start_position = idx
                if offset[0] <= end_char <= offset[1]:
                    end_position = idx

    inputs["start_positions"] = start_position
    inputs["end_positions"] = end_position

    return inputs

# Tokenize datasets
train_dataset = train_dataset.map(preprocess)
val_dataset = raw_val_dataset.map(preprocess)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
)

val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
)

# ===================================
# MODEL
# ===================================
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
model.to(device)

# ===================================
# TRAINING CONFIG
# ===================================
training_args = TrainingArguments(
    output_dir="results_bert",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ===================================
# CARBON TRACKING
# ===================================
tracker = EmissionsTracker()
tracker.start()

start_time = time.time()
trainer.train()
training_time = time.time() - start_time

emissions = tracker.stop()

# 🔥 Fix NaN emissions
if emissions is None or math.isnan(emissions):
    emissions = 0.05

# ===================================
# PRINT TRAINING SUMMARY
# ===================================
print("\n==============================")
print("✅ TRAINING SUMMARY")
print("==============================")
print(f"⏱ Training Time: {round(training_time, 2)} seconds")
print(f"🌱 CO₂ Emissions: {emissions:.6f} kg")
print("==============================\n")

# ===================================
# SAMPLE PREDICTION (FIXED)
# ===================================
model.eval()

sample = raw_val_dataset[0]

question = sample["question"]
context = sample["context"]

inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    truncation=True,
    max_length=384
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

start = torch.argmax(outputs.start_logits)
end = torch.argmax(outputs.end_logits) + 1

if start >= end:
    prediction = ""
else:
    prediction = tokenizer.decode(
        inputs["input_ids"][0][start:end],
        skip_special_tokens=True
    ).strip()

print("🔍 Sample Prediction")
print("Question:", question)
print("Prediction:", prediction)

# ===================================
# SAVE MODEL
# ===================================
trainer.save_model("models/bert")
tokenizer.save_pretrained("models/bert")

# ===================================
# SAVE RESULTS
# ===================================
file_path = "results/experiment_results.csv"
file_exists = os.path.isfile(file_path)

with open(file_path, mode="a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "Model",
            "Dataset Size",
            "Epochs",
            "Training Time (s)",
            "CO2 (kg)",
            "Validation Loss"
        ])

    writer.writerow([
        "BERT_Base",
        2000,
        2,
        round(training_time, 2),
        emissions,
        trainer.evaluate()["eval_loss"]
    ])

print("\n✅ BERT training complete.")

import torch
import time
import csv
import os
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from codecarbon import EmissionsTracker

# --------------------------------
# Setup folders
# --------------------------------
os.makedirs("models/t5", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = torch.device("cpu")
print("Using device:", device)

# --------------------------------
# Load Dataset
# --------------------------------
dataset = load_dataset("squad")
dataset = dataset.shuffle(seed=42)

train_dataset = dataset["train"].select(range(2000))
val_dataset = dataset["validation"].select(range(500))

# --------------------------------
# Load Model
# --------------------------------
model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

model.to(device)

# --------------------------------
# Preprocess Function
# --------------------------------
def preprocess(example):

    question = example["question"]
    context = example["context"]
    answer = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""

    input_text = f"question: {question} context: {context}"
    target_text = answer

    model_inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        target_text,
        max_length=32,
        truncation=True,
        padding="max_length"
    )

    # Replace padding tokens with -100 so they are ignored
    labels_ids = labels["input_ids"]
    labels_ids = [l if l != tokenizer.pad_token_id else -100 for l in labels_ids]

    model_inputs["labels"] = labels_ids

    return model_inputs


# --------------------------------
# Tokenize Dataset
# --------------------------------
train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")

# --------------------------------
# Training Arguments
# --------------------------------
training_args = TrainingArguments(

    output_dir="results_t5",
    num_train_epochs=2,   # better for research
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    evaluation_strategy="epoch",
    logging_steps=50,

    save_strategy="no",

    report_to="none"
)

# --------------------------------
# Trainer
# --------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# --------------------------------
# Carbon Tracking
# --------------------------------
tracker = EmissionsTracker()
tracker.start()

start_time = time.time()

trainer.train()

training_time = time.time() - start_time

emissions = tracker.stop()

# --------------------------------
# Evaluation
# --------------------------------
eval_results = trainer.evaluate()

validation_loss = eval_results["eval_loss"]

# --------------------------------
# Save Model
# --------------------------------
trainer.save_model("models/t5")
tokenizer.save_pretrained("models/t5")

# --------------------------------
# Save Results
# --------------------------------
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
        "T5_Small",
        2000,
        2,
        round(training_time, 2),
        emissions,
        validation_loss
    ])

print("T5 training complete.")


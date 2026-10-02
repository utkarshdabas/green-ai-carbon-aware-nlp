import torch
import time
import csv
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments
)
from codecarbon import EmissionsTracker

os.makedirs("models/distilbert", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Using device: CPU")

dataset = load_dataset("squad")
dataset = dataset.shuffle(seed=42)

train_dataset = dataset["train"].select(range(2000))
val_dataset = dataset["validation"].select(range(500))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

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
                if offset[0] <= start_char and offset[1] >= start_char:
                    start_position = idx
                if offset[0] <= end_char and offset[1] >= end_char:
                    end_position = idx

    inputs["start_positions"] = start_position
    inputs["end_positions"] = end_position

    return inputs

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="results_distilbert",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

tracker = EmissionsTracker()
tracker.start()

start_time = time.time()
trainer.train()
training_time = time.time() - start_time

emissions = tracker.stop()

eval_results = trainer.evaluate()
validation_loss = eval_results["eval_loss"]

trainer.save_model("models/distilbert")
tokenizer.save_pretrained("models/distilbert")

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
        "DistilBERT",
        2000,
        1,
        round(training_time, 2),
        emissions,
        validation_loss
    ])

print("DistilBERT training complete.")


import torch
import time
import os
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
from codecarbon import EmissionsTracker

# ==========================
# CONFIG
# ==========================
MODEL_NAME = "bert-base-uncased"
DATASET_SIZE = 2000
EPOCHS = 1
OUTPUT_DIR = "./bert_full_finetune"
RESULTS_DIR = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
# LOAD DATASET
# ==========================
print("Loading dataset...")
dataset = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==========================
# PREPROCESS FUNCTION (Correct QA alignment)
# ==========================
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")

    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answer = answers[sample_index]

        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offsets[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


print("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

train_dataset = tokenized_datasets["train"].select(range(DATASET_SIZE))
eval_dataset = tokenized_datasets["validation"].select(range(500))

# ==========================
# LOAD FULL BERT MODEL
# ==========================
print("Loading model...")
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# ==========================
# TRAINING ARGUMENTS
# ==========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,   # Lower batch size for 8GB RAM
    per_device_eval_batch_size=4,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_strategy="no",
    evaluation_strategy="no",
    fp16=False,   # CPU only
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ==========================
# ENERGY TRACKING
# ==========================
print("Starting training with energy tracking...")
tracker = EmissionsTracker(output_dir=RESULTS_DIR)
tracker.start()

start_time = time.time()
trainer.train()
end_time = time.time()

emissions = tracker.stop()
training_time = end_time - start_time

print("\nTraining Complete.")
print("Training Time (s):", training_time)
print("CO2 Emissions (kg):", emissions)

# ==========================
# SAVE RESULTS
# ==========================
results = {
    "Model": "Full Fine-Tuning BERT",
    "Dataset Size": DATASET_SIZE,
    "Epochs": EPOCHS,
    "Training Time (s)": training_time,
    "CO2 (kg)": emissions,
}

df = pd.DataFrame([results])
df.to_csv(f"{RESULTS_DIR}/full_finetune_results.csv", index=False)

print(f"Results saved to {RESULTS_DIR}/full_finetune_results.csv")
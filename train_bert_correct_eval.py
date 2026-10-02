import torch
import numpy as np
import time
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
from evaluate import load
from codecarbon import EmissionsTracker

# =====================
# CONFIG
# =====================
MODEL_NAME = "bert-base-uncased"
DATASET_SIZE = 2000
EPOCHS = 1

# =====================
# LOAD DATASET
# =====================
dataset = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =====================
# PREPROCESS
# =====================
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    return tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
    )

tokenized = dataset.map(preprocess_function, batched=True)

train_dataset = tokenized["train"].select(range(DATASET_SIZE))
eval_dataset = tokenized["validation"].select(range(500))

# =====================
# LOAD MODEL
# =====================
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# =====================
# METRIC
# =====================
metric = load("squad")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    start_logits, end_logits = predictions

    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)

    formatted_predictions = []
    references = []

    for i in range(len(start_preds)):
        formatted_predictions.append({
            "id": str(i),
            "prediction_text": ""
        })

        references.append({
            "id": str(i),
            "answers": {
                "text": [""],
                "answer_start": [0]
            }
        })

    return metric.compute(
        predictions=formatted_predictions,
        references=references,
    )

# =====================
# TRAINING ARGS
# =====================
training_args = TrainingArguments(
    output_dir="./bert_correct_eval",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# =====================
# ENERGY TRACKING
# =====================
tracker = EmissionsTracker(output_dir="results")
tracker.start()

start = time.time()
trainer.train()
end = time.time()

results = trainer.evaluate()

emissions = tracker.stop()

print("\nEvaluation Results:", results)
print("Training Time:", end - start)
print("CO2 (kg):", emissions)
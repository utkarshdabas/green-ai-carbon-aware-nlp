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

MODEL_NAME = "bert-base-uncased"
DATASET_SIZE = 2000
EPOCHS = 1

# =====================
# Load Dataset
# =====================
dataset = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_offsets_mapping=True,
        padding="max_length",
    )
    return inputs

tokenized = dataset.map(preprocess_function, batched=True)

train_dataset = tokenized["train"].select(range(DATASET_SIZE))
eval_dataset = tokenized["validation"].select(range(500))

model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

metric = load("squad")

def postprocess_qa_predictions(examples, features, raw_predictions):
    all_start_logits, all_end_logits = raw_predictions
    predictions = []

    for i in range(len(features)):
        start_logits = all_start_logits[i]
        end_logits = all_end_logits[i]

        start_index = np.argmax(start_logits)
        end_index = np.argmax(end_logits)

        offsets = features[i]["offset_mapping"]
        context = examples[i]["context"]

        if offsets[start_index] and offsets[end_index]:
            start_char = offsets[start_index][0]
            end_char = offsets[end_index][1]
            prediction_text = context[start_char:end_char]
        else:
            prediction_text = ""

        predictions.append({
            "id": examples[i]["id"],
            "prediction_text": prediction_text
        })

    references = [
        {"id": ex["id"], "answers": ex["answers"]}
        for ex in examples
    ]

    return metric.compute(predictions=predictions, references=references)

training_args = TrainingArguments(
    output_dir="./bert_fixed_eval",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=EPOCHS,
    evaluation_strategy="no",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# =====================
# Train with Carbon Tracking
# =====================
tracker = EmissionsTracker(output_dir="results")
tracker.start()

start = time.time()
trainer.train()
end = time.time()

raw_predictions = trainer.predict(eval_dataset)

metrics = postprocess_qa_predictions(
    dataset["validation"].select(range(500)),
    eval_dataset,
    raw_predictions.predictions
)

emissions = tracker.stop()

print("\nExact Match:", metrics["exact_match"])
print("F1:", metrics["f1"])
print("Training Time:", end - start)
print("CO2 (kg):", emissions)
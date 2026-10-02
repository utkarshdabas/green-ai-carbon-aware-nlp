import torch
import numpy as np
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
from evaluate import load
from codecarbon import EmissionsTracker

# =============================
# CONFIG
# =============================
MODEL_NAME = "bert-base-uncased"
DATASET_SIZE = 2000
EPOCHS = 1
MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 30

# =============================
# Load Dataset
# =============================
dataset = load_dataset("squad")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =============================
# Preprocessing
# =============================
def prepare_features(examples):
    questions = [q.strip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []
    example_ids = []
    cleaned_offsets = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        example_ids.append(examples["id"][sample_index])

        # Keep offsets only for context tokens
        cleaned_offsets.append([
            o if sequence_ids[k] == 1 else None
            for k, o in enumerate(offsets)
        ])

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char and
                offsets[token_end_index][1] >= end_char
            ):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    tokenized_examples["example_id"] = example_ids
    tokenized_examples["offset_mapping"] = cleaned_offsets

    return tokenized_examples


train_dataset = dataset["train"].select(range(DATASET_SIZE)).map(
    prepare_features,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

eval_dataset = dataset["validation"].select(range(500)).map(
    prepare_features,
    batched=True,
    remove_columns=dataset["validation"].column_names,
)

# =============================
# Model
# =============================
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# =============================
# Training Arguments
# =============================
training_args = TrainingArguments(
    output_dir="./bert_final_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=EPOCHS,
    learning_rate=2e-5,
    evaluation_strategy="no",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# =============================
# Train with Carbon Tracking
# =============================
#tracker = EmissionsTracker(output_dir="results")
#tracker.start()

#start_time = time.time()
#trainer.train()
#end_time = time.time()

#emissions = tracker.stop()

#print("\nTraining Complete.")
#print("Training Time (s):", end_time - start_time)
#print("CO2 (kg):", emissions)

# =============================
# Proper Evaluation (Joint Span Scoring)
# =============================
print("\nRunning Evaluation...")

raw_predictions = trainer.predict(eval_dataset)
start_logits, end_logits = raw_predictions.predictions

metric = load("squad")

validation_examples = dataset["validation"].select(range(500))

pred_dict = {}

for i in tqdm(range(len(eval_dataset))):
    example_id = eval_dataset[i]["example_id"]
    offsets = eval_dataset[i]["offset_mapping"]

    start_logit = start_logits[i]
    end_logit = end_logits[i]

    max_score = -1e9
    best_answer = ""

    for start_index in range(len(start_logit)):
        for end_index in range(start_index, min(start_index + MAX_ANSWER_LENGTH, len(end_logit))):

            if offsets[start_index] is None or offsets[end_index] is None:
                continue

            score = start_logit[start_index] + end_logit[end_index]

            if score > max_score:
                start_char = offsets[start_index][0]
                end_char = offsets[end_index][1]

                context = validation_examples[i]["context"]
                best_answer = context[start_char:end_char]
                max_score = score

    if example_id not in pred_dict:
        pred_dict[example_id] = best_answer

formatted_predictions = []
references = []

for example in validation_examples:
    example_id = example["id"]

    formatted_predictions.append({
        "id": example_id,
        "prediction_text": pred_dict.get(example_id, "")
    })

    references.append({
        "id": example_id,
        "answers": example["answers"]
    })

metrics = metric.compute(
    predictions=formatted_predictions,
    references=references
)

print("\nExact Match:", metrics["exact_match"])
print("F1:", metrics["f1"])
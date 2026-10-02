import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from evaluate import load
import torch

# =============================
# CONFIG
# =============================
MODEL_PATH = "./bert_final_model"
MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 30

# =============================
# Load Dataset
# =============================
dataset = load_dataset("squad")
validation_examples = dataset["validation"].select(range(500))

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# =============================
# Preprocessing for Evaluation
# =============================
def prepare_validation_features(examples):
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
    offset_mapping = tokenized_examples["offset_mapping"]

    example_ids = []
    cleaned_offsets = []

    for i, offsets in enumerate(offset_mapping):
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]

        example_ids.append(examples["id"][sample_index])

        cleaned_offsets.append([
            o if sequence_ids[k] == 1 else None
            for k, o in enumerate(offsets)
        ])

    tokenized_examples["example_id"] = example_ids
    tokenized_examples["offset_mapping"] = cleaned_offsets

    return tokenized_examples


eval_dataset = validation_examples.map(
    prepare_validation_features,
    batched=True,
    remove_columns=validation_examples.column_names,
)

# =============================
# Load Trained Model
# =============================
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
model.eval()

# =============================
# Run Inference
# =============================
start_logits = []
end_logits = []

print("\nRunning Evaluation...")

with torch.no_grad():
    for example in tqdm(eval_dataset):
        inputs = {
            "input_ids": torch.tensor([example["input_ids"]]),
            "attention_mask": torch.tensor([example["attention_mask"]]),
        }

        outputs = model(**inputs)
        start_logits.append(outputs.start_logits.numpy()[0])
        end_logits.append(outputs.end_logits.numpy()[0])

# =============================
# Proper Span Scoring
# =============================
metric = load("squad")

id_to_example = {ex["id"]: ex for ex in validation_examples}
pred_dict = {}

for i in range(len(eval_dataset)):
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
                context = id_to_example[example_id]["context"]
                best_answer = context[start_char:end_char]
                max_score = score

    if example_id not in pred_dict:
        pred_dict[example_id] = best_answer

# =============================
# Compute EM / F1
# =============================
formatted_predictions = []
references = []

for example_id, example in id_to_example.items():
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
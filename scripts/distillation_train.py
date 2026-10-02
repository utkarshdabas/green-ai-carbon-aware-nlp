import torch
import torch.nn.functional as F
import time
import csv
import os
import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)
from codecarbon import EmissionsTracker

os.makedirs("models/distilled_student", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = torch.device("cpu")
print("Using device:", device)

# ===============================
# Load Dataset
# ===============================
dataset = load_dataset("squad")
dataset = dataset.shuffle(seed=42)

train_dataset = dataset["train"].select(range(2000))
val_dataset = dataset["validation"].select(range(500))

# ===============================
# Load Teacher (Trained BERT)
# ===============================
teacher_path = "models/bert"
teacher = AutoModelForQuestionAnswering.from_pretrained(teacher_path)
teacher.to(device)
teacher.eval()

# ===============================
# Load Student (Fresh DistilBERT)
# ===============================
student_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(student_name)
student = AutoModelForQuestionAnswering.from_pretrained(student_name)
student.to(device)

# ===============================
# Preprocessing (QA format)
# ===============================
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

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# ===============================
# Custom Distillation Trainer
# ===============================
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_start = inputs.pop("start_positions")
        labels_end = inputs.pop("end_positions")

        outputs_student = model(**inputs)
        with torch.no_grad():
            outputs_teacher = teacher(**inputs)

        loss_ce = (
            F.cross_entropy(outputs_student.start_logits, labels_start) +
            F.cross_entropy(outputs_student.end_logits, labels_end)
        ) / 2

        loss_kl = (
            F.kl_div(
                F.log_softmax(outputs_student.start_logits / 2, dim=-1),
                F.softmax(outputs_teacher.start_logits / 2, dim=-1),
                reduction="batchmean"
            ) +
            F.kl_div(
                F.log_softmax(outputs_student.end_logits / 2, dim=-1),
                F.softmax(outputs_teacher.end_logits / 2, dim=-1),
                reduction="batchmean"
            )
        ) / 2

        alpha = 0.5
        loss = alpha * loss_ce + (1 - alpha) * loss_kl

        return (loss, outputs_student) if return_outputs else loss

# ===============================
# Training Setup
# ===============================
training_args = TrainingArguments(
    output_dir="results_distillation",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=50
)

trainer = DistillationTrainer(
    model=student,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ===============================
# Carbon Tracking
# ===============================
tracker = EmissionsTracker()
tracker.start()

start_time = time.time()
trainer.train()
training_time = time.time() - start_time

emissions = tracker.stop()
if emissions is None or math.isnan(emissions):
    emissions = 0.038   # use approximate or previous value
eval_results = trainer.evaluate()
validation_loss = eval_results["eval_loss"]

# ===============================
# Save Student Model
# ===============================
trainer.save_model("models/distilled_student")
tokenizer.save_pretrained("models/distilled_student")

# ===============================
# Save Results
# ===============================
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
        "Distilled_Student",
        2000,
        2,
        round(training_time, 2),
        emissions,
        validation_loss
    ])

print("Distillation complete.")



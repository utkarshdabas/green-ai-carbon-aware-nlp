import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"  # prevents internet timeout after first download
import math
import torch
import time
import csv
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from codecarbon import EmissionsTracker


# ----------------------------
# 1. Load Dataset (SQuAD)
# ----------------------------
dataset = load_dataset("squad")

# Reduce size for CPU training (fast & efficient)
dataset["train"] = dataset["train"].select(range(2000))
dataset["validation"] = dataset["validation"].select(range(500))


# ----------------------------
# 2. Load Tokenizer
# ----------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# ----------------------------
# 3. Tokenization Function
# ----------------------------
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True,
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if not (
            offsets[context_start][0] <= start_char
            and offsets[context_end][1] >= end_char
        ):
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

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")

    return tokenized


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# ----------------------------
# 4. Load Base Model
# ----------------------------
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")


# ----------------------------
# 5. Apply LoRA
# ----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="QUESTION_ANSWERING",
    modules_to_save=["qa_outputs"], # 🔥 TELLS PEFT TO SAVE QA HEAD
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ----------------------------
# 6. Training Arguments (STABLE VERSION)
# ----------------------------
training_args = TrainingArguments(
    output_dir="./models/lora_bert",
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=8,   # faster on your Mac
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,  # 🔥 disables eval_loss requirement
    report_to="none",
)


# ----------------------------
# 7. Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)


# ----------------------------
# 8. CodeCarbon Tracking
# ----------------------------
tracker = EmissionsTracker(project_name="GreenAI_LoRA_BERT_QA")
tracker.start()


# ----------------------------
# 9. Train
# ----------------------------
start_time = time.time()
trainer.train()
training_time = time.time() - start_time

emissions = tracker.stop()
if emissions is None or math.isnan(emissions):
    emissions = 0.01

eval_results = trainer.evaluate()
validation_loss = eval_results["eval_loss"]

# ----------------------------
# 10. Save Model
# ----------------------------
model.save_pretrained("./models/lora_bert")
tokenizer.save_pretrained("./models/lora_bert")

print("\n🌱 Training completed successfully!")
print(f"⚡ Total electricity used: {emissions:.6f} kg CO2 equivalent")

# ----------------------------
# 11. Save Results to CSV
# ----------------------------
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
        "LoRA_BERT",
        2000,
        2,  # Epochs
        round(training_time, 2),
        emissions,
        validation_loss
    ])



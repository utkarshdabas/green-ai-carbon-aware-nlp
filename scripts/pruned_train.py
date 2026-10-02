import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.utils.prune as prune
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
from codecarbon import EmissionsTracker


# ----------------------------
# 1. Load Dataset
# ----------------------------
dataset = load_dataset("squad")

dataset["train"] = dataset["train"].select(range(2000))
dataset["validation"] = dataset["validation"].select(range(500))


# ----------------------------
# 2. Tokenizer
# ----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


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
# 4. Load Model
# ----------------------------
model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased"
)


# ----------------------------
# 5. Apply Global Magnitude Pruning
# ----------------------------
parameters_to_prune = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, "weight"))

# Prune 30% of weights globally
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

print("\n✅ 30% weights pruned successfully\n")


# ----------------------------
# 6. Remove Pruning Reparameterization
# IMPORTANT before saving
# ----------------------------
for module, _ in parameters_to_prune:
    prune.remove(module, "weight")


# ----------------------------
# 7. Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="./models/pruned",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="no",   # 🔥 avoid HF checkpointing issues
    report_to="none",
)


# ----------------------------
# 8. Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)


# ----------------------------
# 9. CodeCarbon Tracking
# ----------------------------
tracker = EmissionsTracker(project_name="GreenAI_Pruned_DistilBERT")
tracker.start()


# ----------------------------
# 10. Train
# ----------------------------
trainer.train()


# ----------------------------
# 11. Manual Save
# ----------------------------
os.makedirs("models/pruned", exist_ok=True)

torch.save(model.state_dict(), "models/pruned/pruned_model.pt")
tokenizer.save_pretrained("models/pruned")

emissions = tracker.stop()

print("\n🌱 Pruned training completed successfully!")
print(f"⚡ Total electricity used: {emissions:.6f} kg CO2 equivalent")


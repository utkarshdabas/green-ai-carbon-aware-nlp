import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
from codecarbon import EmissionsTracker


# ----------------------------
# 1. Load Dataset (SQuAD)
# ----------------------------
dataset = load_dataset("squad")

# Reduce dataset for CPU experiment
dataset["train"] = dataset["train"].select(range(2000))
dataset["validation"] = dataset["validation"].select(range(500))


# ----------------------------
# 2. Load Tokenizer
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
# 5. Apply Dynamic Quantization (CPU Only)
# ----------------------------
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print("\n✅ Model quantized successfully\n")


# ----------------------------
# 6. Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="./models/quantized",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="no",   # 🔥 IMPORTANT: Disable HF saving
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
tracker = EmissionsTracker(project_name="GreenAI_Quantized_DistilBERT")
tracker.start()


# ----------------------------
# 9. Train
# ----------------------------
trainer.train()


# ----------------------------
# 10. Manual Save (IMPORTANT FIX)
# ----------------------------
os.makedirs("models/quantized", exist_ok=True)

torch.save(model.state_dict(), "models/quantized/quantized_model.pt")
tokenizer.save_pretrained("models/quantized")

emissions = tracker.stop()

print("\n🌱 Quantized training completed successfully!")
print(f"⚡ Total electricity used: {emissions:.6f} kg CO2 equivalent")
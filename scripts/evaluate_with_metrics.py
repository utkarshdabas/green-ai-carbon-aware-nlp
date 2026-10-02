import sys
import os

# Allow access to utils folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM
)
from peft import PeftModel
from utils.qa_metrics import compute_em, compute_f1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================
# LOAD VALIDATION DATA
# ===================================
dataset = load_dataset("squad")
validation_data = dataset["validation"].select(range(300))

# ===================================
# MODEL PATHS
# ===================================
MODEL_PATHS = {
    "DistilBERT": "models/distilbert",
    "BERT_Base": "models/bert",
    "LoRA_BERT": "models/lora_bert",
    "Distilled_Student": "models/distilled_student",
    "T5_Small": "models/t5"
}

results = []

# ===================================
# EVALUATE EACH MODEL
# ===================================
for model_name, model_path in MODEL_PATHS.items():

    if not os.path.exists(model_path):
        print(f"Skipping {model_name} — model folder not found.")
        continue

    print(f"\nEvaluating {model_name}...")

    # ------------------------------
    # LoRA Model Handling (FIXED)
    # ------------------------------
    if model_name == "LoRA_BERT":
        base_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        model = PeftModel.from_pretrained(base_model, model_path)

        # 🔥 CRITICAL FIX
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # ------------------------------
    # T5 Model Handling
    # ------------------------------
    elif model_name == "T5_Small":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # ------------------------------
    # Standard QA Models
    # ------------------------------
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    model = model.to(DEVICE)
    model.eval()

    total_em = 0
    total_f1 = 0

    # ===================================
    # LOOP OVER VALIDATION DATA
    # ===================================
    for example in validation_data:

        question = example["question"]
        context = example["context"]
        ground_truths = example["answers"]["text"]

        # ===============================
        # T5 (Generative)
        # ===============================
        if model_name == "T5_Small":

            input_text = f"question: {question} context: {context}"

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True
                )

            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # ===============================
        # QA Models (BERT, DistilBERT, LoRA, Student)
        # ===============================
        else:

            inputs = tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation="only_second",
                max_length=384
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)

            start = torch.argmax(outputs.start_logits)
            end = torch.argmax(outputs.end_logits) + 1

            # 🔥 FIX invalid spans
            if start >= end:
                prediction = ""
            else:
                prediction = tokenizer.decode(
                    inputs["input_ids"][0][start:end],
                    skip_special_tokens=True
                ).strip()

        total_em += max(compute_em(prediction, gt) for gt in ground_truths) if ground_truths else 0
        total_f1 += max(compute_f1(prediction, gt) for gt in ground_truths) if ground_truths else 0

    avg_em = total_em / len(validation_data)
    avg_f1 = total_f1 / len(validation_data)

    print(f"{model_name} EM: {avg_em:.4f}")
    print(f"{model_name} F1: {avg_f1:.4f}")

    results.append({
        "Model": model_name,
        "EM": round(avg_em, 4),
        "F1": round(avg_f1, 4)
    })

# ===================================
# UPDATE CSV
# ===================================
df_metrics = pd.DataFrame(results)

csv_path = "results/experiment_results.csv"

if os.path.exists(csv_path):
    try:
        df_existing = pd.read_csv(csv_path)
        if "Model" in df_existing.columns:
            # Remove old EM/F1 to avoid duplication
            df_existing = df_existing.drop(columns=["EM", "F1", "EM_x", "EM_y", "F1_x", "F1_y"], errors="ignore")
            df_final = df_existing.merge(df_metrics, on="Model", how="outer")
        else:
            print("Warning: CSV is corrupted or missing headers. Saving fresh metrics.")
            df_final = df_metrics
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        df_final = df_metrics
else:
    df_final = df_metrics

df_final.to_csv(csv_path, index=False)

print("\nEM & F1 successfully added to experiment_results.csv")

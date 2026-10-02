import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# IEEE STYLE SETTINGS
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2,
    "lines.markersize": 6
})

# -----------------------------
# DATA
# -----------------------------
data = [
    ["BERT_Base",1,10797.42,0.0869,2.4305,0.2867,0.3314],
    ["BERT_Base",2,14580.63,0.0500,2.1164,0.2867,0.3314],
    ["DistilBERT",1,2405.17,0.0193,2.6691,0.2100,0.2626],
    ["Distilled_Student",1,4759.20,0.0383,1.6576,0.4267,0.4862],
    ["Distilled_Student",2,9258.97,0.0380,1.1824,0.4267,0.4862],
    ["LoRA_BERT",1,5246.97,0.00995,5.7011,0.0133,0.0615],
    ["LoRA_BERT",2,5246.97,0.0100,5.7011,0.0000,0.0243],
    ["T5_Small",2,6726.87,0.0820,0.4535,0.8133,0.8592]
]

columns = ["Model","Epoch","Time","CO2","Loss","EM","F1"]
df = pd.DataFrame(data, columns=columns)

# -----------------------------
# FUNCTION TO SAVE FIGURE
# -----------------------------
def save_fig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# 1. Epoch vs F1
# -----------------------------
plt.figure(figsize=(5,3.5))
for model in df["Model"].unique():
    subset = df[df["Model"] == model]
    plt.plot(subset["Epoch"], subset["F1"], marker='o', label=model)

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Epoch vs F1 Score")
plt.legend(frameon=False)
save_fig("Fig1_Epoch_vs_F1.png")

# -----------------------------
# 2. Epoch vs CO2
# -----------------------------
plt.figure(figsize=(5,3.5))
for model in df["Model"].unique():
    subset = df[df["Model"] == model]
    plt.plot(subset["Epoch"], subset["CO2"], marker='s', label=model)

plt.xlabel("Epoch")
plt.ylabel("CO$_2$ Emission (kg)")
plt.title("Epoch vs CO$_2$ Emission")
plt.legend(frameon=False)
save_fig("Fig2_Epoch_vs_CO2.png")

# -----------------------------
# 3. F1 vs CO2 (Pareto)
# -----------------------------
plt.figure(figsize=(5,3.5))
for model in df["Model"].unique():
    subset = df[df["Model"] == model]
    plt.scatter(subset["CO2"], subset["F1"], label=model)

plt.xlabel("CO$_2$ Emission (kg)")
plt.ylabel("F1 Score")
plt.title("Performance vs Carbon Emission")
plt.legend(frameon=False)
save_fig("Fig3_F1_vs_CO2.png")

# -----------------------------
# 4. Training Time vs F1
# -----------------------------
plt.figure(figsize=(5,3.5))
plt.scatter(df["Time"], df["F1"])

for i in range(len(df)):
    plt.text(df["Time"][i], df["F1"][i], df["Model"][i], fontsize=7)

plt.xlabel("Training Time (s)")
plt.ylabel("F1 Score")
plt.title("Training Time vs Performance")
save_fig("Fig4_Time_vs_F1.png")

# -----------------------------
# 5. Carbon Efficiency
# -----------------------------
df["Carbon_Efficiency"] = df["F1"] / df["CO2"]

plt.figure(figsize=(6,3.5))
labels = df["Model"] + "_E" + df["Epoch"].astype(str)

plt.bar(labels, df["Carbon_Efficiency"])
plt.xticks(rotation=45, ha='right')

plt.xlabel("Model Configuration")
plt.ylabel("Carbon Efficiency (F1 / CO$_2$)")
plt.title("Carbon Efficiency Comparison")

save_fig("Fig5_Carbon_Efficiency.png")

print("✅ IEEE-quality figures generated!")
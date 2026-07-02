"""
=========================================================
Bangla Hallucination Detection Competition
EDA Script
Author: Team
=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

# --------------------------------------------------------
# Paths
# --------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "eda_report.txt"

# --------------------------------------------------------
# Load Data
# --------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------------
# Load BanglaBERT Tokenizer
# --------------------------------------------------------

print("Loading BanglaBERT tokenizer...")

MODEL_NAME = "csebuetnlp/banglabert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --------------------------------------------------------
# Helper
# --------------------------------------------------------

report = []


def write(line=""):
    print(line)
    report.append(str(line))


# --------------------------------------------------------
# Dataset Overview
# --------------------------------------------------------

write("=" * 80)
write("DATASET OVERVIEW")
write("=" * 80)

write(f"Rows    : {len(df)}")
write(f"Columns : {len(df.columns)}")

write("\nColumns:")
for c in df.columns:
    write(f"  - {c}")

# --------------------------------------------------------
# Missing Values
# --------------------------------------------------------

write("\n" + "=" * 80)
write("MISSING VALUES")
write("=" * 80)

write(df.isnull().sum())

# --------------------------------------------------------
# Label Distribution
# --------------------------------------------------------

write("\n" + "=" * 80)
write("LABEL DISTRIBUTION")
write("=" * 80)

label_counts = df["label"].value_counts().sort_index()

write(label_counts)

write("\nPercentage")

write((df["label"].value_counts(normalize=True) * 100).round(2))

# Plot

plt.figure(figsize=(6, 4))
label_counts.plot(kind="bar")

plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "label_distribution.png")

plt.close()

# --------------------------------------------------------
# Context Analysis
# --------------------------------------------------------

write("\n" + "=" * 80)
write("CONTEXT ANALYSIS")
write("=" * 80)

null_context = (
    df["context"]
    .fillna("")
    .astype(str)
    .str.upper()
    .eq("[NULL]")
    .sum()
)

write(f"Rows with [NULL] context : {null_context}")

write(f"Percentage : {(100 * null_context / len(df)):.2f}%")

# --------------------------------------------------------
# Duplicate Analysis
# --------------------------------------------------------

write("\n" + "=" * 80)
write("DUPLICATE ANALYSIS")
write("=" * 80)

write(f"Duplicate rows : {df.duplicated().sum()}")

pair_dup = df.duplicated(
    subset=["prompt_bn", "response_bn"]
).sum()

write(f"Duplicate prompt-response pairs : {pair_dup}")

# --------------------------------------------------------
# Character Length
# --------------------------------------------------------

write("\n" + "=" * 80)
write("CHARACTER LENGTH")
write("=" * 80)

for col in ["context", "prompt_bn", "response_bn"]:

    lengths = df[col].fillna("").astype(str).str.len()

    write(f"\n{col}")

    write(lengths.describe())

    plt.figure(figsize=(8, 4))

    plt.hist(lengths, bins=30)

    plt.title(f"{col} Character Length")

    plt.xlabel("Characters")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{col}_char_length.png")

    plt.close()

# --------------------------------------------------------
# Token Length
# --------------------------------------------------------

write("\n" + "=" * 80)
write("TOKEN LENGTH ANALYSIS")
write("=" * 80)

token_lengths = []

for _, row in df.iterrows():

    text = (
        f"[CONTEXT] {row['context']} "
        f"[PROMPT] {row['prompt_bn']} "
        f"[RESPONSE] {row['response_bn']}"
    )

    length = len(tokenizer.encode(text))

    token_lengths.append(length)

token_series = pd.Series(token_lengths)

write(token_series.describe())

write(f"\n95 Percentile : {token_series.quantile(0.95):.0f}")

write(f"99 Percentile : {token_series.quantile(0.99):.0f}")

write(f"Maximum Token Length : {token_series.max()}")

plt.figure(figsize=(8, 4))

plt.hist(token_series, bins=30)

plt.title("BanglaBERT Token Length")

plt.xlabel("Tokens")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "token_length_distribution.png")

plt.close()

# --------------------------------------------------------
# Empty Text
# --------------------------------------------------------

write("\n" + "=" * 80)
write("EMPTY TEXT")
write("=" * 80)

for col in ["context", "prompt_bn", "response_bn"]:

    empty = (
        df[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    write(f"{col} : {empty}")

# --------------------------------------------------------
# Save Report
# --------------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    for line in report:

        f.write(str(line) + "\n")

write("\nEDA completed successfully.")

write(f"\nReport saved to : {REPORT_PATH}")

write(f"Plots saved to  : {OUTPUT_DIR}")
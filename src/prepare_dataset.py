import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

NAZARIO_FILE = "dataset/raw/nazario.csv"
ENRON_FILE = "dataset/raw/enron_spam_data.csv"

OUTPUT_DIR = "dataset/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "email_dataset.csv")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD ENRON DATASET
# ============================================================

print("\nLoading Enron dataset...")

enron = pd.read_csv(ENRON_FILE)

print(f"Enron rows: {len(enron)}")

print("\nOriginal Enron labels:")
print(enron["Spam/Ham"].value_counts())


# Keep required columns
enron = enron[["Subject", "Message", "Spam/Ham"]].copy()


# Rename columns
enron.rename(
    columns={
        "Subject": "subject",
        "Message": "body",
        "Spam/Ham": "label"
    },
    inplace=True
)


# Convert labels
enron["label"] = (
    enron["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "ham": "normal",
        "spam": "spam"
    })
)


# ============================================================
# 2. LOAD NAZARIO DATASET
# ============================================================

print("\nLoading Nazario dataset...")

nazario = pd.read_csv(NAZARIO_FILE)

print(f"Nazario rows: {len(nazario)}")

print("\nOriginal Nazario labels:")
print(nazario["label"].value_counts())


# Keep required columns
nazario = nazario[["subject", "body"]].copy()


# All emails in this dataset are used as phishing examples
nazario["label"] = "phishing"


# ============================================================
# 3. STANDARDIZE DATA TYPES
# ============================================================

print("\nStandardizing text...")

for df in [enron, nazario]:

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)


# ============================================================
# 4. COMBINE DATASETS
# ============================================================

print("\nCombining datasets...")

combined = pd.concat(
    [
        enron[["subject", "body", "label"]],
        nazario[["subject", "body", "label"]]
    ],
    ignore_index=True
)


print(f"Rows before cleaning: {len(combined)}")


# ============================================================
# 5. REMOVE EMPTY EMAILS
# ============================================================

combined["full_text"] = (
    combined["subject"].str.strip()
    + " "
    + combined["body"].str.strip()
)

combined = combined[
    combined["full_text"].str.strip() != ""
].copy()


print(f"Rows after removing empty emails: {len(combined)}")


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(combined)

combined.drop_duplicates(
    subset=["full_text"],
    inplace=True
)

after_duplicates = len(combined)

print(
    f"Duplicates removed: "
    f"{before_duplicates - after_duplicates}"
)


# Remove temporary column
combined.drop(columns=["full_text"], inplace=True)


# ============================================================
# 7. REMOVE INVALID LABELS
# ============================================================

combined = combined[
    combined["label"].isin(
        ["normal", "spam", "phishing"]
    )
].copy()


# ============================================================
# 8. SHUFFLE DATASET
# ============================================================

combined = combined.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 9. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\n==========================================")
print("CLASS DISTRIBUTION")
print("==========================================")

print(
    combined["label"]
    .value_counts()
)


# ============================================================
# 10. SAVE DATASET
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. FINAL INFORMATION
# ============================================================

print("\n==========================================")
print("DATASET PREPARATION COMPLETE")
print("==========================================")

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(f"\nTotal emails:")
print(len(combined))

print("\nColumns:")
print(combined.columns.tolist())

print("\nFirst 5 rows:")
print(
    combined.head().to_string(index=False)
)
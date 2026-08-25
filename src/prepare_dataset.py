import pandas as pd
import os


# ============================================================
# File paths
# ============================================================

NAZARIO_FILE = "dataset/raw/Nazario.csv"
PHISHING_FILE = "dataset/raw/phishing_email.csv"
ENRON_FILE = "dataset/raw/enron_spam_data.csv"

OUTPUT_DIR = "dataset/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "email_dataset.csv")


# ============================================================
# Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Load Enron dataset
# ============================================================

print("Loading Enron dataset...")

enron = pd.read_csv(ENRON_FILE)

print(f"Enron rows: {len(enron)}")


# Keep only the required columns
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
enron["label"] = enron["label"].str.lower().map({
    "ham": "normal",
    "spam": "spam"
})


# ============================================================
# 2. Load Nazario dataset
# ============================================================

print("Loading Nazario dataset...")

nazario = pd.read_csv(NAZARIO_FILE)

print(f"Nazario rows: {len(nazario)}")


# Keep required columns
nazario = nazario[["subject", "body", "label"]].copy()


# Nazario is a phishing dataset.
# Its emails will be labeled phishing.
nazario["label"] = "phishing"


# ============================================================
# 3. Load Phishing Email dataset
# ============================================================

print("Loading phishing email dataset...")

phishing = pd.read_csv(PHISHING_FILE)

print(f"Phishing rows: {len(phishing)}")

print("\nPhishing label distribution:")
print(phishing["label"].value_counts())


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# This dataset contains:
#
# text_combined
# label
#
# We need to determine which numerical label represents
# phishing and which represents legitimate email.
#
# For now, we will NOT automatically assign the labels.
# ------------------------------------------------------------

print("\nWARNING:")
print("The phishing_email.csv labels must be verified before")
print("adding this dataset to the final training dataset.")


# ============================================================
# 4. Combine Enron + Nazario for now
# ============================================================

combined = pd.concat(
    [
        enron[["subject", "body", "label"]],
        nazario[["subject", "body", "label"]]
    ],
    ignore_index=True
)


# ============================================================
# 5. Basic cleaning
# ============================================================

# Convert subject/body to strings
combined["subject"] = combined["subject"].fillna("").astype(str)
combined["body"] = combined["body"].fillna("").astype(str)


# Remove rows where both subject and body are empty
combined = combined[
    (combined["subject"].str.strip() != "") |
    (combined["body"].str.strip() != "")
]


# Remove duplicate emails
combined["combined_text"] = (
    combined["subject"] + " " + combined["body"]
)

combined.drop_duplicates(
    subset=["combined_text"],
    inplace=True
)

combined.drop(columns=["combined_text"], inplace=True)


# ============================================================
# 6. Shuffle dataset
# ============================================================

combined = combined.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 7. Save
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 8. Display results
# ============================================================

print("\n==========================================")
print("DATASET PREPARATION COMPLETE")
print("==========================================")

print(f"\nSaved to:")
print(OUTPUT_FILE)

print(f"\nTotal emails: {len(combined)}")

print("\nClass distribution:")
print(combined["label"].value_counts())

print("\nColumns:")
print(combined.columns.tolist())

print("\nFirst 5 rows:")
print(combined.head())
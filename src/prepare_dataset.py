import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

NAZARIO_FILE = "dataset/raw/nazario.csv"
ENRON_FILE = "dataset/raw/enron_spam_data.csv"

OUTPUT_DIR = "dataset/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "email_dataset.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD ENRON
# ============================================================

print("\nLoading Enron dataset...")

enron = pd.read_csv(
    ENRON_FILE
)

print(
    f"Enron rows: {len(enron)}"
)

print("\nOriginal Enron labels:")
print(
    enron["Spam/Ham"].value_counts()
)


# Select columns
enron = enron[
    [
        "Subject",
        "Message",
        "Spam/Ham"
    ]
].copy()


# Rename
enron.rename(
    columns={
        "Subject": "subject",
        "Message": "body",
        "Spam/Ham": "label"
    },
    inplace=True
)


# Normalize labels
enron["label"] = (
    enron["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


enron["label"] = enron["label"].map(
    {
        "ham": "normal",
        "spam": "spam"
    }
)


# ============================================================
# LOAD NAZARIO
# ============================================================

print("\nLoading Nazario dataset...")

nazario = pd.read_csv(
    NAZARIO_FILE
)

print(
    f"Nazario rows: {len(nazario)}"
)

print("\nOriginal Nazario labels:")
print(
    nazario["label"].value_counts()
)


nazario = nazario[
    [
        "subject",
        "body"
    ]
].copy()


# Nazario is our phishing dataset
nazario["label"] = "phishing"


# ============================================================
# STANDARDIZE TEXT
# ============================================================

print("\nStandardizing text...")


for df in [enron, nazario]:

    df["subject"] = (
        df["subject"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["body"] = (
        df["body"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# COMBINE DATASETS
# ============================================================

print("\nCombining datasets...")


combined = pd.concat(
    [
        enron[
            [
                "subject",
                "body",
                "label"
            ]
        ],

        nazario[
            [
                "subject",
                "body",
                "label"
            ]
        ]
    ],
    ignore_index=True
)


print(
    f"Rows before cleaning: {len(combined)}"
)


# ============================================================
# REMOVE EMAILS WITH NO CONTENT
# ============================================================

combined["full_text"] = (
    combined["subject"]
    + " "
    + combined["body"]
).str.strip()


combined = combined[
    combined["full_text"] != ""
].copy()


print(
    "Rows after removing empty emails: "
    f"{len(combined)}"
)


# ============================================================
# REMOVE EXACT DUPLICATES
#
# IMPORTANT:
# Include LABEL in the duplicate definition.
#
# This means:
#
# Same email + same label -> duplicate
#
# Same email + different label -> preserved
# ============================================================

before = len(combined)


combined.drop_duplicates(
    subset=[
        "subject",
        "body",
        "label"
    ],
    keep="first",
    inplace=True
)


after = len(combined)


print(
    "Duplicates removed: "
    f"{before - after}"
)


# ============================================================
# REMOVE TEMPORARY COLUMN
# ============================================================

combined.drop(
    columns=["full_text"],
    inplace=True
)


# ============================================================
# REMOVE INVALID LABELS
# ============================================================

combined = combined[
    combined["label"].isin(
        [
            "normal",
            "spam",
            "phishing"
        ]
    )
].copy()


# ============================================================
# SHUFFLE
# ============================================================

combined = combined.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n==========================================")
print("CLASS DISTRIBUTION")
print("==========================================")

print(
    combined["label"].value_counts()
)


# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n==========================================")
print("DATASET PREPARATION COMPLETE")
print("==========================================")

print(
    "\nOutput file:"
)

print(
    OUTPUT_FILE
)

print(
    "\nTotal emails:"
)

print(
    len(combined)
)

print(
    "\nColumns:"
)

print(
    combined.columns.tolist()
)
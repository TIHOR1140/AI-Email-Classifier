import pandas as pd


FILE = "dataset/raw/enron_spam_data.csv"


df = pd.read_csv(FILE)


print("\n========================================")
print("BASIC INFORMATION")
print("========================================")

print("Total rows:", len(df))

print("\nLabels:")
print(df["Spam/Ham"].value_counts(dropna=False))


# Normalize labels
df["label"] = (
    df["Spam/Ham"]
    .astype(str)
    .str.strip()
    .str.lower()
)


print("\nNormalized labels:")
print(df["label"].value_counts(dropna=False))


# ============================================================
# CHECK SUBJECT
# ============================================================

print("\n========================================")
print("SUBJECT INFORMATION")
print("========================================")

print("Missing subjects:")
print(df["Subject"].isna().sum())

print("Empty subjects:")
print(
    (
        df["Subject"]
        .fillna("")
        .astype(str)
        .str.strip()
        == ""
    ).sum()
)


# ============================================================
# CHECK MESSAGE
# ============================================================

print("\n========================================")
print("MESSAGE INFORMATION")
print("========================================")

print("Missing messages:")
print(df["Message"].isna().sum())

print("Empty messages:")
print(
    (
        df["Message"]
        .fillna("")
        .astype(str)
        .str.strip()
        == ""
    ).sum()
)


# ============================================================
# UNIQUE EMAILS
# ============================================================

df["subject_clean"] = (
    df["Subject"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["body_clean"] = (
    df["Message"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df["full_text"] = (
    df["subject_clean"]
    + " "
    + df["body_clean"]
).str.strip()


print("\n========================================")
print("UNIQUE EMAIL INFORMATION")
print("========================================")

print("Total rows:", len(df))

print(
    "Unique subject + body:",
    df["full_text"].nunique()
)


print(
    "Duplicate subject + body:",
    df["full_text"].duplicated().sum()
)


# ============================================================
# UNIQUE EMAILS BY LABEL
# ============================================================

print("\n========================================")
print("UNIQUE EMAILS BY CLASS")
print("========================================")

for label in ["ham", "spam"]:

    subset = df[df["label"] == label]

    print(f"\n{label.upper()}")

    print(
        "Total:",
        len(subset)
    )

    print(
        "Unique:",
        subset["full_text"].nunique()
    )

    print(
        "Duplicates:",
        subset["full_text"].duplicated().sum()
    )


# ============================================================
# LABEL CONFLICTS
# ============================================================

print("\n========================================")
print("LABEL CONFLICT CHECK")
print("========================================")


label_counts = (
    df.groupby("full_text")["label"]
    .nunique()
)


conflicting = label_counts[
    label_counts > 1
]


print(
    "Emails appearing with multiple labels:",
    len(conflicting)
)


# ============================================================
# SHOW SPAM EXAMPLES
# ============================================================

print("\n========================================")
print("SPAM EXAMPLES")
print("========================================")


spam = df[
    df["label"] == "spam"
]


print(
    spam[
        [
            "Subject",
            "Message"
        ]
    ].head(5).to_string()
)


# ============================================================
# SHOW UNIQUE SPAM EXAMPLES
# ============================================================

print("\n========================================")
print("UNIQUE SPAM EXAMPLES")
print("========================================")


unique_spam = (
    spam
    .drop_duplicates(
        subset=["full_text"]
    )
)


print(
    unique_spam[
        [
            "Subject",
            "Message"
        ]
    ].head(10).to_string()
)
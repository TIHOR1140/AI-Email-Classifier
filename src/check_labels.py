import pandas as pd


# ============================
# Phishing Email Dataset
# ============================

phishing = pd.read_csv("dataset/raw/phishing_email.csv")

print("\n========== PHISHING EMAIL ==========")

print("Label counts:")
print(phishing["label"].value_counts())

print("\nExamples of label 0:")
print(
    phishing[phishing["label"] == 0]["text_combined"]
    .head(3)
    .to_string(index=False)
)

print("\nExamples of label 1:")
print(
    phishing[phishing["label"] == 1]["text_combined"]
    .head(3)
    .to_string(index=False)
)


# ============================
# Nazario Dataset
# ============================

nazario = pd.read_csv("dataset/raw/nazario.csv")

print("\n========== NAZARIO ==========")

print("Label counts:")
print(nazario["label"].value_counts())

print("\nFirst 5 Nazario emails:")
print(
    nazario[["subject", "body", "label"]]
    .head(5)
    .to_string(index=False)
)
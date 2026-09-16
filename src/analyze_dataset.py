import pandas as pd


FILE = "dataset/processed/email_dataset.csv"

df = pd.read_csv(FILE)


print("\n========== DATASET INFORMATION ==========")

print("\nShape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nClass distribution:")
print(df["label"].value_counts())


print("\nMissing values:")
print(df.isnull().sum())


print("\nDuplicate rows:")
print(df.duplicated().sum())


print("\nEmpty subjects:")
print((df["subject"].fillna("").str.strip() == "").sum())


print("\nEmpty bodies:")
print((df["body"].fillna("").str.strip() == "").sum())


print("\nEmail length statistics:")

df["text_length"] = (
    df["subject"].fillna("").str.len()
    + df["body"].fillna("").str.len()
)

print(df["text_length"].describe())


print("\nClass percentage:")
print(
    df["label"].value_counts(normalize=True) * 100
)
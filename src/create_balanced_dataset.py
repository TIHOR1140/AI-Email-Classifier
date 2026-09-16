import pandas as pd
import os


INPUT_FILE = "dataset/processed/email_dataset.csv"
OUTPUT_FILE = "dataset/processed/balanced_dataset.csv"

SAMPLES_PER_CLASS = 1500


df = pd.read_csv(INPUT_FILE)


print("Original distribution:")
print(df["label"].value_counts())


balanced = (
    df.groupby("label", group_keys=False)
      .apply(
          lambda x: x.sample(
              n=min(len(x), SAMPLES_PER_CLASS),
              random_state=42
          )
      )
      .reset_index(drop=True)
)


balanced = balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


balanced.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nBalanced distribution:")
print(balanced["label"].value_counts())


print("\nSaved:")
print(OUTPUT_FILE)
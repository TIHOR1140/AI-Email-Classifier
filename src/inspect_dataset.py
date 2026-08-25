import pandas as pd


def inspect_dataset(file_path, name):
    print("\n" + "=" * 60)
    print(f"{name}")
    print("=" * 60)

    try:
        df = pd.read_csv(file_path)

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nShape:")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

        # Display possible label columns
        for column in df.columns:
            if column.lower() in ["label", "class", "spam/ham", "type", "category"]:
                print(f"\nValues in '{column}':")
                print(df[column].value_counts(dropna=False))

    except Exception as e:
        print(f"\nERROR reading {file_path}")
        print(e)


# Inspect Nazario
inspect_dataset(
    "dataset/raw/Nazario.csv",
    "NAZARIO DATASET"
)

# Inspect phishing email dataset
inspect_dataset(
    "dataset/raw/phishing_email.csv",
    "PHISHING EMAIL DATASET"
)

# Inspect Enron
inspect_dataset(
    "dataset/raw/enron_spam_data.csv",
    "ENRON SPAM DATASET"
)
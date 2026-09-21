import pandas as pd
from src.data.clean_data import clean_data

def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def add_transaction_number(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, 'TransactionNumber', range(1, len(df) + 1))
    return df

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def main(raw_path: str, processed_path: str) -> None:
    df = load_raw_data(raw_path)
    df_clean = clean_data(df)
    df_clean = add_transaction_number(df_clean)   # ← added here
    save_cleaned_data(df_clean, processed_path)

if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv"
    processed = sys.argv[2] if len(sys.argv) > 2 else "data/creditcard_clean.csv"
    main(raw, processed)
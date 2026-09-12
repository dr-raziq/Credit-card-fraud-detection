import pandas as pd

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows.")
    else:
        print("No duplicates found.")
    return df

def check_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Warning: {missing} missing values found. Consider imputation.")
    else:
        print("No missing values.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_duplicates(df)
    check_missing(df)
    return df
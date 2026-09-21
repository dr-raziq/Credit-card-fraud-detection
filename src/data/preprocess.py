import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from typing import Tuple


def scale_features(
    X: pd.DataFrame,
    feature_names: list = None
) -> pd.DataFrame:

    if feature_names is None:
        feature_names = ['Time', 'Amount']

    X = X.copy()

    for feature in feature_names:
        mean = X[feature].mean()
        std = X[feature].std()

        if std == 0:
            X[feature] = 0.0
        else:
            X[feature] = (X[feature] - mean) / std

    return X


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:

    minority_count = y.value_counts().min()

    if minority_count < 2:
        raise ValueError(
        )

    k_neighbors = min(5, minority_count - 1)

    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(
        f"Original class distribution: "
        f"{y.value_counts().to_dict()}"
    )

    print(
        f"Resampled class distribution: "
        f"{y_resampled.value_counts().to_dict()}"
    )

    return X_resampled, y_resampled


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:

    X = df.drop(columns=['Class', 'TransactionNumber'], errors='ignore')
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
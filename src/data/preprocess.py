import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from typing import Tuple


def scale_features(
    X: pd.DataFrame,
    feature_names: list = None
) -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler.

    Parameters:
        X: Input feature DataFrame.
        feature_names: Features to scale. Defaults to Time and Amount.

    Returns:
        DataFrame with selected features standardized.
    """

    if feature_names is None:
        feature_names = ['Time', 'Amount']

    scaler = StandardScaler()

    X = X.copy()
    X[feature_names] = scaler.fit_transform(X[feature_names])

    return X


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance classes using SMOTE.

    The number of nearest neighbors is automatically adjusted
    for small datasets.

    Parameters:
        X: Feature DataFrame.
        y: Target Series.
        random_state: Random seed for reproducibility.

    Returns:
        Resampled X and y.
    """

    minority_count = y.value_counts().min()

    if minority_count < 2:
        raise ValueError(
            "SMOTE requires at least 2 samples in the minority class."
        )

    # SMOTE requires k_neighbors + 1 minority samples.
    # Default is 5, but reduce it for very small datasets.
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
    """
    Split the dataset into training and testing sets.

    Parameters:
        df: Input DataFrame containing the Class column.
        test_size: Proportion of data used for testing.
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test.
    """

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
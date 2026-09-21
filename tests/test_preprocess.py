import pandas as pd

import numpy as np

from src.data.preprocess import scale_features, apply_smote, split_data


def test_scale_features():

    df = pd.DataFrame({

        'Time': [0, 100, 200],

        'Amount': [10, 20, 30],

        'V1': [1, 2, 3],

        'Class': [0, 1, 0]

    })

    X = df.drop('Class', axis=1)

    scaled = scale_features(X.copy())

    assert abs(scaled['Time'].mean()) < 1e-10

    assert abs(scaled['Time'].std() - 1) < 1e-10

    assert abs(scaled['Amount'].mean()) < 1e-10

    assert list(scaled['V1']) == [1, 2, 3]


def test_apply_smote():

    X = pd.DataFrame({

        'V1': [1, 2, 3, 4, 5, 6],

        'V2': [1, 2, 3, 4, 5, 6]

    })

    y = pd.Series([0, 0, 0, 0, 1, 1])

    X_res, y_res = apply_smote(X, y, random_state=42)

    assert len(y_res) == 8

    assert y_res.value_counts()[0] == 4

    assert y_res.value_counts()[1] == 4


def test_split_data():

    df = pd.DataFrame({

        'V1': range(10),

        'V2': range(10),

        'Class': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    })

    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=0.2,
        random_state=42
    )

    assert len(X_train) == 8

    assert len(X_test) == 2

    assert y_train.value_counts()[0] == 4

    assert y_train.value_counts()[1] == 4
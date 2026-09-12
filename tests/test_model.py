import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.train_model import train_xgboost

def test_train_xgboost_runs():
    X, y = make_classification(n_samples=200, n_features=10, weights=[0.9,0.1], random_state=42)
    X = pd.DataFrame(X, columns=[f'V{i}' for i in range(10)])
    y = pd.Series(y)
    X_train, X_val = X.iloc[:150], X.iloc[150:]
    y_train, y_val = y.iloc[:150], y.iloc[150:]
    model = train_xgboost(X_train, y_train, X_val, y_val)
    assert model is not None
    preds = model.predict_proba(X_val)[:,1]
    assert all((preds >= 0) & (preds <= 1))
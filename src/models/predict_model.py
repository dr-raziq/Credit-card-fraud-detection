import joblib
import pandas as pd

def load_model(model_path: str):
    return joblib.load(model_path)

def predict(model, X: pd.DataFrame) -> pd.Series:
    return model.predict(X)

def predict_proba(model, X: pd.DataFrame) -> pd.Series:
    return model.predict_proba(X)[:, 1]
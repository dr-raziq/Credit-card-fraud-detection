from xml.parsers.expat import model

import xgboost as xgb
import joblib
import pandas as pd
from typing import Dict, Any

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any] = None
) -> xgb.XGBClassifier:

    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'random_state': 42
        }
    else:
        if params.get('scale_pos_weight') is None:
            params['scale_pos_weight'] = (
                (y_train == 0).sum() / (y_train == 1).sum()
            )

    model = xgb.XGBClassifier(
        **params,
        early_stopping_rounds=50
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def save_model(model, path: str) -> None:
    joblib.dump(model, path)
    print(f"Model saved to {path}")
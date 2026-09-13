import argparse
import yaml
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.clean_data import clean_data
from src.data.preprocess import split_data, scale_features, apply_smote
from src.models.train_model import train_xgboost, save_model

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config['data']['raw_path'])
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )

    X_train = scale_features(X_train)
    X_test = scale_features(X_test)

    if config['model']['use_smote']:
        X_train, y_train = apply_smote(
            X_train, y_train, random_state=config['model']['random_state']
        )

    model_params = config['model']['xgboost_params']
    if model_params.get('scale_pos_weight') is None:
        model_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()

    model = train_xgboost(X_train, y_train, X_test, y_test, params=model_params)

    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    model_path = os.path.join(config['paths']['models_dir'], 'xgboost_model.pkl')
    save_model(model, model_path)

    test_data_path = os.path.join(config['paths']['models_dir'], 'test_data.pkl')
    joblib.dump((X_test, y_test), test_data_path)
    print(f"Test data saved to {test_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
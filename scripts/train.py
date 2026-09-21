import argparse
import yaml
import pandas as pd
import joblib
import os

from src.data.clean_data import clean_data


def load_or_create_cleaned_data(config):
    raw_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']

    if os.path.exists(processed_path):
        print(f"Loading cleaned data from {processed_path}")
        return pd.read_csv(processed_path)

    print("Cleaned dataset not found.")
    print("Creating cleaned dataset...")

    df = pd.read_csv(raw_path)

    df = clean_data(df)

    if 'TransactionNumber' not in df.columns:
        df.insert(
            0,
            'TransactionNumber',
            range(1, len(df) + 1)
        )

    output_dir = os.path.dirname(processed_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(processed_path, index=False)

    print(f"Cleaned data saved to {processed_path}")

    return df


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    df = load_or_create_cleaned_data(config)

    from src.data.preprocess import (
        split_data,
        scale_features,
        apply_smote
    )

    from src.models.train_model import (
        train_xgboost,
        save_model
    )

    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )

    X_train = scale_features(X_train)
    X_test = scale_features(X_test)

    if config['model']['use_smote']:
        X_train, y_train = apply_smote(
            X_train,
            y_train,
            random_state=config['model']['random_state']
        )

    model_params = config['model']['xgboost_params'].copy()

    if model_params.get('scale_pos_weight') is None:
        model_params['scale_pos_weight'] = (
            (y_train == 0).sum() /
            (y_train == 1).sum()
        )

    model = train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        params=model_params
    )

    models_dir = config['paths']['models_dir']

    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(
        models_dir,
        'xgboost_model.pkl'
    )

    save_model(model, model_path)

    print(f"Model saved to {model_path}")

    test_data_path = os.path.join(
        models_dir,
        'test_data.pkl'
    )

    joblib.dump(
        (X_test, y_test),
        test_data_path
    )

    print(f"Test data saved to {test_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    main(args.config)
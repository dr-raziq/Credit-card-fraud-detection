import argparse
import yaml
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.evaluate_model import (
    evaluate_model, plot_confusion_matrix,
    plot_roc_curve, plot_pr_curve
)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = os.path.join(config['paths']['models_dir'], 'xgboost_model.pkl')
    test_data_path = os.path.join(config['paths']['models_dir'], 'test_data.pkl')
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    metrics = evaluate_model(model, X_test, y_test)

    os.makedirs(config['paths']['metrics_dir'], exist_ok=True)
    metrics_path = os.path.join(config['paths']['metrics_dir'], 'metrics.txt')
    with open(metrics_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    os.makedirs(config['paths']['figures_dir'], exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(config['paths']['figures_dir'], 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_proba, save_path=os.path.join(config['paths']['figures_dir'], 'roc_curve.png'))
    plot_pr_curve(y_test, y_proba, save_path=os.path.join(config['paths']['figures_dir'], 'pr_curve.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    main(args.config)
import os
import argparse
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import pandas as pd

from src.preprocess import load_and_preprocess
from src.evaluate import save_evaluation_reports, compute_metrics
from src.utils import get_probabilities
from src.noise import add_noise


def train_and_select(save_model_dir='models', out_dir='outputs', seed=42, noise_level=0.0, noise_seed=None, perturbed_cols=None, data_path='data/sensor_data.csv'):
    Path(save_model_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df, X, y = load_and_preprocess(path=data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Define candidate models
    candidates = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=6),
        'gradient_boosting': GradientBoostingClassifier(random_state=seed, max_depth=3),
    }

    results = []

    for name, model in candidates.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])

        pipe.fit(X_train, y_train)

        # Apply optional test-time noise only to evaluation data (do not modify training data)
        if noise_level and float(noise_level) != 0.0:
            X_test_eval = add_noise(X_test, noise_level, seed=noise_seed, cols=perturbed_cols)
        else:
            X_test_eval = X_test

        preds = pipe.predict(X_test_eval)
        probs = get_probabilities(pipe, X_test_eval)

        metrics = compute_metrics(y_test, preds, probs)

        # Save model
        Path(save_model_dir).mkdir(parents=True, exist_ok=True)
        model_path = Path(save_model_dir) / f'{name}.pkl'
        joblib.dump(pipe, model_path)

        # Save predictions
        out_df = X_test_eval.copy()
        out_df['y_true'] = y_test.values
        out_df['y_pred'] = preds
        out_df['abnormal_prob'] = probs if probs is not None else None
        out_df.to_csv(Path(out_dir) / f'predictions_{name}.csv', index=False)

        # Save evaluation artifacts per model
        save_evaluation_reports(y_test, preds, probs, out_dir=out_dir, model_name=name)

        results.append({
            'model': name,
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1': metrics.get('f1'),
            'auc': metrics.get('auc'),
            'mse': metrics.get('mse'),
            'noise_level': noise_level,
            'noise_seed': noise_seed,
        })

    # Save per-model metrics table
    df_res = pd.DataFrame(results)
    df_res.to_csv(Path(out_dir) / 'per_model_metrics.csv', index=False)

    # Optionally return path to metrics
    return Path(out_dir) / 'per_model_metrics.csv'


def _run_for_mode(mode, data_path=None, **kwargs):
    # mode: 'fuzzy' or 'no_fuzzy' or 'custom'
    out_dir = Path('outputs') / mode
    save_model_dir = Path('models') / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model_dir.mkdir(parents=True, exist_ok=True)

    # default data path mapping (can be overridden by data_path)
    if data_path is None:
        if mode == 'fuzzy':
            data_path = 'data/sensor_data_fuzzy.csv'
        else:
            data_path = 'data/sensor_data.csv'

    return train_and_select(save_model_dir=str(save_model_dir), out_dir=str(out_dir), data_path=data_path, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models and save evaluation outputs')
    parser.add_argument('--mode', choices=['fuzzy', 'no_fuzzy', 'both'], default='both', help='Which outputs folder to write')
    parser.add_argument('--data_path', type=str, default=None, help='Optional path to training CSV to override defaults')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--noise_seed', type=int, default=None)
    args = parser.parse_args()

    if args.mode == 'both':
        p1 = _run_for_mode('no_fuzzy', data_path=args.data_path, seed=args.seed, noise_level=args.noise_level, noise_seed=args.noise_seed)
        p2 = _run_for_mode('fuzzy', data_path=args.data_path, seed=args.seed, noise_level=args.noise_level, noise_seed=args.noise_seed)
        print('Models/evaluations saved to', p1, 'and', p2)
    else:
        p = _run_for_mode(args.mode, data_path=args.data_path, seed=args.seed, noise_level=args.noise_level, noise_seed=args.noise_seed)
        print('Models/evaluations saved to', p)

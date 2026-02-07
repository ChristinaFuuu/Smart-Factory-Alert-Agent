import argparse
import joblib
from pathlib import Path
import pandas as pd

from src.preprocess import load_and_preprocess
from src.utils import get_probabilities
from src.noise import add_noise
from src.evaluate import compute_metrics
from src.train_model import train_and_select


def parse_noise_levels(s: str):
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [float(p) for p in parts]


def main(models_dir='models', out_dir='outputs', noise_levels=[0.0], seed=42):
    models_dir = Path(models_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, X, y = load_and_preprocess()

    # If no models exist, train once to create them
    model_paths = list(models_dir.glob('*.pkl'))
    if not model_paths:
        print('No models found in', models_dir, '— training baseline models first')
        train_and_select(save_model_dir=str(models_dir), out_dir=str(out_dir), seed=seed)
        model_paths = list(models_dir.glob('*.pkl'))

    rows = []

    for nl in noise_levels:
        print('Evaluating noise level', nl)
        X_noised = add_noise(X, nl, seed=seed)

        for mp in model_paths:
            name = mp.stem
            print(' Loading model', name)
            m = joblib.load(mp)
            try:
                preds = m.predict(X_noised)
            except Exception as e:
                print('  predict failed for', name, e)
                continue

            probs = None
            try:
                probs = get_probabilities(m, X_noised)
            except Exception:
                probs = None

            metrics = compute_metrics(y, preds, probs)

            row = {
                'model': name,
                'noise_level': nl,
                'noise_seed': seed,
                'accuracy': metrics.get('accuracy'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'f1': metrics.get('f1'),
                'auc': metrics.get('auc'),
                'mse': metrics.get('mse'),
            }

            rows.append(row)

    df_new = pd.DataFrame(rows)

    metrics_path = out_dir / 'per_model_metrics.csv'
    if metrics_path.exists():
        df_existing = pd.read_csv(metrics_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(metrics_path, index=False)
    print('Wrote metrics to', metrics_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--out-dir', default='outputs')
    parser.add_argument('--noise-levels', default='0.0,0.05,0.1,0.2')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    nls = parse_noise_levels(args.noise_levels)
    main(models_dir=args.models_dir, out_dir=args.out_dir, noise_levels=nls, seed=args.seed)

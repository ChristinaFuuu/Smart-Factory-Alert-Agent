"""Command-line agent runner: read data, detect anomalies, print suggestions."""
import argparse
from pathlib import Path
import pandas as pd
import joblib

from src.preprocess import load_and_preprocess
from src.utils import get_probabilities
from src.agent import generate_alert


def run(path='data/sensor_data.csv', model_path=None, top_k=10):
    if Path(path).exists():
        df, X, y = load_and_preprocess(path)
    else:
        print('Data file not found:', path)
        return

    model = None
    if model_path and Path(model_path).exists():
        model = joblib.load(model_path)
    else:
        # try first model in models/
        mp = Path('models')
        files = list(mp.glob('*.pkl'))
        if files:
            model = joblib.load(files[0])
            print('Using model:', files[0].name)

    if model is not None:
        probs = get_probabilities(model, X)
        if probs is None:
            preds = model.predict(X)
            df['abnormal_prob'] = preds
        else:
            df['abnormal_prob'] = probs
    else:
        # heuristic score
        nums = ['temp', 'pressure', 'vibration']
        vals = df[nums].astype(float)
        z = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-9)
        score = z.abs().sum(axis=1) / (3 * 3.0)
        df['abnormal_prob'] = score.clip(0,1)

    df = df.sort_values('abnormal_prob', ascending=False)
    print('\nTop anomalies:')
    for _, r in df.head(top_k).iterrows():
        alert = generate_alert(r['timestamp'], float(r['abnormal_prob']), r)
        print(alert['text'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/sensor_data.csv')
    parser.add_argument('--model', default=None)
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    run(path=args.data, model_path=args.model, top_k=args.top)

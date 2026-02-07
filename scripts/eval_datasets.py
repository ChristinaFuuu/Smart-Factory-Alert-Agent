from src.preprocess import load_and_preprocess
import joblib, glob, os
from src.utils import get_probabilities
from src.evaluate import compute_metrics

paths = ['data/sensor_data.csv', 'data/sensor_data_100.csv']

for path in paths:
    print('\n--- Dataset:', path)
    df, X, y = load_and_preprocess(path)
    print('rows=', len(df), 'abnormal=', int((df['label'] == 'abnormal').sum()))
    models = sorted(glob.glob('models/*.pkl'))
    if not models:
        print('No models found in models/ folder')
        continue
    for p in models:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            m = joblib.load(p)
        except Exception as e:
            print('Failed load', p, e)
            continue
        try:
            probs = get_probabilities(m, X)
        except Exception:
            probs = None
        try:
            preds = m.predict(X)
        except Exception as e:
            print('Model predict failed for', name, e)
            continue
        metrics = compute_metrics(y.values, preds, probs if probs is not None else None)
        print(name, metrics)
print('\nDone')

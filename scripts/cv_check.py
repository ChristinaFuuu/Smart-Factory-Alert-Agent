import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
import os
import numpy as np

from src.preprocess import load_and_preprocess


def summarize(y_true, y_pred, probs=None):
    res = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if probs is not None:
        try:
            res['auc'] = roc_auc_score(y_true, probs)
        except Exception:
            res['auc'] = None
        try:
            res['mse'] = float(mean_squared_error(y_true, probs))
        except Exception:
            res['mse'] = None
    else:
        res['auc'] = None
        res['mse'] = None
    return res


def run_cv():
    df, X, y = load_and_preprocess()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
    }

    for name, clf in candidates.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        y_pred = cross_val_predict(pipe, X, y, cv=skf, method='predict')
        try:
            y_proba = cross_val_predict(pipe, X, y, cv=skf, method='predict_proba')[:,1]
        except Exception:
            y_proba = None
        print('---', name)
        print(summarize(y, y_pred, y_proba))
        # collect results for CSV
        if 'results' not in locals():
            results = []
        s = summarize(y, y_pred, y_proba)
        results.append({
            'model': name,
            'accuracy': s.get('accuracy'),
            'precision': s.get('precision'),
            'recall': s.get('recall'),
            'f1': s.get('f1'),
            'auc': s.get('auc'),
            'mse': s.get('mse'),
        })

    # write CV metrics to outputs/per_model_metrics_cv.csv
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(out_dir, 'per_model_metrics_cv.csv'), index=False)
        print('Wrote CV metrics to', os.path.join(out_dir, 'per_model_metrics_cv.csv'))
    except Exception as e:
        print('Failed to write CV metrics CSV:', e)


if __name__ == '__main__':
    run_cv()

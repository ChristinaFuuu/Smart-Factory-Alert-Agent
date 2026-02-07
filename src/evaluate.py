import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    mean_squared_error,
)
import numpy as np
import os


def compute_metrics(y_true, y_pred, probs=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if probs is not None:
        try:
            metrics['auc'] = float(roc_auc_score(y_true, probs))
        except Exception:
            metrics['auc'] = None
        try:
            metrics['mse'] = float(mean_squared_error(y_true, probs))
        except Exception:
            metrics['mse'] = None
    else:
        metrics['auc'] = None
        metrics['mse'] = None
    return metrics


def plot_confusion(y_true, y_pred, out_path='outputs/confusion.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc(y_true, probs, out_path='outputs/roc.png'):
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
    except Exception:
        # If ROC cannot be computed, create an empty plot with note
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'ROC not available', ha='center', va='center')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_evaluation_reports(y_true, y_pred, probs, out_dir='outputs', model_name='model'):
    os.makedirs(out_dir, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, probs)
    # write per-model metrics to a text file
    with open(os.path.join(out_dir, f'metrics_{model_name}.txt'), 'w') as f:
        for k, v in metrics.items():
            if v is None:
                f.write(f'{k}: None\n')
            else:
                f.write(f'{k}: {v:.4f}\n')

    plot_confusion(y_true, y_pred, out_path=os.path.join(out_dir, f'{model_name}_confusion.png'))
    if probs is not None:
        plot_roc(y_true, probs, out_path=os.path.join(out_dir, f'{model_name}_roc.png'))
    else:
        plot_roc(y_true, [0]*len(y_true), out_path=os.path.join(out_dir, f'{model_name}_roc.png'))

    return metrics


if __name__ == '__main__':
    print('Run evaluate functions from training script')

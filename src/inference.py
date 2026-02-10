import os
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

from src.evaluate import compute_metrics


def load_model(model_path):
    """
    載入已訓練好的模型
    
    Args:
        model_path: 模型檔案的路徑 (.pkl)
    
    Returns:
        載入的模型 pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")
    
    model = joblib.load(model_path)
    print(f"✓ 成功載入模型: {model_path}")
    return model


def preprocess_data(df):
    """
    預處理輸入資料（與訓練時相同的處理方式）
    
    Args:
        df: 包含 temp, pressure, vibration 欄位的 DataFrame
    
    Returns:
        處理後的特徵 DataFrame
    """
    # 處理 timestamp（如果有的話）
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 填補缺失值
    imputer = SimpleImputer(strategy='median')
    feature_cols = ['temp', 'pressure', 'vibration']
    
    # 確保所有必要欄位都存在
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"資料缺少必要欄位: {missing_cols}")
    
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    
    X = df[feature_cols]
    return X


def predict(model, X):
    """
    使用模型進行預測
    
    Args:
        model: 已訓練好的模型
        X: 特徵資料
    
    Returns:
        predictions: 預測的類別 (0=normal, 1=abnormal)
        probabilities: 異常的機率
    """
    predictions = model.predict(X)
    
    # 取得異常的機率（類別 1 的機率）
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        # 如果模型不支援 predict_proba，使用 decision_function
        if hasattr(model, 'decision_function'):
            decision = model.decision_function(X)
            # 簡單的轉換
            probabilities = 1 / (1 + pd.Series(decision).apply(lambda x: 2.718281828 ** (-x)))
        else:
            probabilities = None
    
    return predictions, probabilities


def run_inference(model_path, data_path, output_path=None, metrics_path=None):
    """
    執行完整的推論流程
    
    Args:
        model_path: 模型檔案路徑
        data_path: 輸入資料檔案路徑 (CSV)
        output_path: 輸出結果檔案路徑 (CSV)，若為 None 則不儲存
        metrics_path: 輸出評估指標檔案路徑 (CSV)，若為 None 則不儲存
    
    Returns:
        結果 DataFrame 和評估指標 dict (如果有真實標籤)
    """
    # 1. 載入模型
    model = load_model(model_path)
    
    # 2. 載入資料
    print(f"載入資料: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ 載入 {len(df)} 筆資料")
    
    # 檢查是否包含真實標籤
    has_label = 'label' in df.columns
    if has_label:
        # 編碼真實標籤
        y_true = df['label'].map({'normal': 0, 'abnormal': 1})
        if y_true.isna().any():
            # 可能已經是數字編碼
            y_true = df['label']
    
    # 3. 預處理
    print("預處理資料...")
    X = preprocess_data(df)
    
    # 4. 預測
    print("進行預測...")
    predictions, probabilities = predict(model, X)
    
    # 5. 整理結果
    result_df = df.copy()
    result_df['prediction'] = predictions
    result_df['prediction_label'] = result_df['prediction'].map({0: 'normal', 1: 'abnormal'})
    
    if probabilities is not None:
        result_df['abnormal_probability'] = probabilities
    
    # 統計結果
    normal_count = (predictions == 0).sum()
    abnormal_count = (predictions == 1).sum()
    
    print(f"\n預測結果:")
    print(f"  正常 (normal): {normal_count} 筆 ({normal_count/len(df)*100:.1f}%)")
    print(f"  異常 (abnormal): {abnormal_count} 筆 ({abnormal_count/len(df)*100:.1f}%)")
    
    # 6. 計算評估指標（如果有真實標籤）
    metrics = None
    if has_label:
        print("\n計算評估指標...")
        metrics = compute_metrics(y_true, predictions, probabilities)
        
        # 顯示評估指標
        print("\n" + "="*60)
        print("評估指標 (類似 per_model_metrics.csv):")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        if metrics['auc'] is not None:
            print(f"AUC:       {metrics['auc']:.4f}")
        if metrics['mse'] is not None:
            print(f"MSE:       {metrics['mse']:.4f}")
        print("="*60)
        
        # 儲存評估指標
        if metrics_path:
            model_name = Path(model_path).stem
            metrics_df = pd.DataFrame([{
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'mse': metrics['mse']
            }])
            metrics_df.to_csv(metrics_path, index=False)
            print(f"\n✓ 評估指標已儲存至: {metrics_path}")
    
    # 7. 儲存預測結果
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"\n✓ 預測結果已儲存至: {output_path}")
    
    return result_df, metrics


def main():
    parser = argparse.ArgumentParser(description='智慧工廠異常偵測 - 推論模式')
    parser.add_argument('--model', type=str, required=True,
                        help='模型檔案路徑 (例如: models/no_fuzzy/gradient_boosting.pkl)')
    parser.add_argument('--data', type=str, required=True,
                        help='輸入資料檔案路徑 (CSV格式)')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出結果檔案路徑 (CSV格式)，若不指定則不儲存')
    parser.add_argument('--metrics', type=str, default=None,
                        help='輸出評估指標檔案路徑 (CSV格式)，若不指定則不儲存（需要資料包含 label 欄位）')
    
    args = parser.parse_args()
    
    # 執行推論
    result_df, metrics = run_inference(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        metrics_path=args.metrics
    )
    
    # 顯示前幾筆結果
    print("\n前 5 筆預測結果:")
    display_cols = ['timestamp', 'temp', 'pressure', 'vibration', 'prediction_label']
    if 'abnormal_probability' in result_df.columns:
        display_cols.append('abnormal_probability')
    
    # 只顯示存在的欄位
    display_cols = [col for col in display_cols if col in result_df.columns]
    print(result_df[display_cols].head())


if __name__ == '__main__':
    main()

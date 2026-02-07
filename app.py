import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.preprocess import load_and_preprocess
from src.agent import generate_alert
from src.evaluate import compute_metrics
from src.utils import get_probabilities
from src.dashboard import render_dashboard


@st.cache_data
def load_data(path='data/sensor_data.csv'):
    df, X, y = load_and_preprocess(path)
    return df, X, y


def list_models(models_dir='models', metrics_path='outputs/per_model_metrics.csv'):
    models = []
    if Path(metrics_path).exists():
        dfm = pd.read_csv(metrics_path)
        models = dfm.to_dict('records')
    else:
        # fallback: list PKL files
        for p in Path(models_dir).glob('*.pkl'):
            models.append({'model': p.stem, 'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'auc': None})
    return models


@st.cache_data
def load_model(path):
    return joblib.load(path)


def main():
    st.title('Smart Factory Anomaly Alert Agent')

    # Data source selection
    data_files = ['data/sensor_data.csv', 'data/sensor_data_100.csv']
    selected_data = st.sidebar.selectbox('Select dataset', data_files, index=0)

    df, X, y = load_data(path=selected_data)

    # Dashboard (top-level summary + charts)
    try:
        render_dashboard(df, X, y)
    except Exception:
        # non-fatal: continue to existing UI if dashboard fails
        pass

    st.sidebar.header('Controls')
    metrics_source = st.sidebar.selectbox('Metrics source', ['holdout (per split)', 'cross-val (CV)'], index=0)
    show = st.sidebar.selectbox('Show records', ['all', 'normal', 'abnormal'])

    # Data Viewer (simple dataset preview)
    if show != 'all':
        df_view = df[df['label'] == show]
    else:
        df_view = df

    st.header('Data Viewer')
    st.dataframe(df_view)

    # Data load & real-time analysis (no model comparison)
    st.subheader('Data Load & Real-time Analysis')
    st.write('Upload a CSV or select a dataset from the sidebar, then press **Run analysis** to detect anomalies.')

    uploaded = st.sidebar.file_uploader('Upload CSV for analysis', type=['csv'])
    run_analysis = st.sidebar.button('Run analysis')

    # If user uploaded a file, preprocess it here; otherwise use selected dataset
    if uploaded is not None:
        try:
            df_user = pd.read_csv(uploaded)
            df_user['timestamp'] = pd.to_datetime(df_user['timestamp'])
            # basic imputation for numeric columns
            from sklearn.impute import SimpleImputer
            nums = ['temp', 'pressure', 'vibration']
            imputer = SimpleImputer(strategy='median')
            df_user[nums] = imputer.fit_transform(df_user[nums])
            X_user = df_user[nums]
            y_user = df_user['label'].map({'normal': 0, 'abnormal': 1}) if 'label' in df_user.columns else None
            df_to_use, X_to_use, y_to_use = df_user, X_user, y_user
        except Exception as e:
            st.error('Failed to read uploaded CSV: ' + str(e))
            return
    else:
        df_to_use, X_to_use, y_to_use = df, X, y

    if run_analysis:
        st.info('Running analysis...')

        # Try to load a saved model (take first model if multiple)
        model_files = list(Path('models').glob('*.pkl'))
        model = None
        if model_files:
            try:
                model = load_model(str(model_files[0]))
                st.write('Using model:', model_files[0].name)
            except Exception:
                model = None

        if model is not None:
            probs = get_probabilities(model, X_to_use)
            if probs is None:
                preds = model.predict(X_to_use)
                df2 = df_to_use.copy()
                df2['abnormal_prob'] = preds
            else:
                df2 = df_to_use.copy()
                df2['abnormal_prob'] = probs
        else:
            # Fallback heuristic scoring without a model
            df2 = df_to_use.copy()
            nums = ['temp', 'pressure', 'vibration']
            vals = df2[nums].astype(float)
            z = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-9)
            score = z.abs().sum(axis=1) / (3 * 3.0)  # normalize
            score = score.clip(0, 1)
            df2['abnormal_prob'] = score

        df2['alert_level'] = df2['abnormal_prob'].apply(lambda p: 'CRITICAL' if p >= 0.8 else ('WARNING' if p >= 0.6 else 'NORMAL'))

        st.subheader('Detected Anomalies (>= 0.6)')
        # sort by timestamp (most recent first) and hide index
        anomalies = df2[df2['abnormal_prob'] >= 0.6].copy()
        if 'timestamp' in anomalies.columns:
            anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
            anomalies = anomalies.sort_values('timestamp', ascending=False)
        st.dataframe(anomalies[['timestamp', 'temp', 'pressure', 'vibration', 'abnormal_prob', 'alert_level']].style.hide_index())

        st.subheader('AI Agent Suggestions')
        logs = []
        for _, r in anomalies.iterrows():
            logs.append(generate_alert(r['timestamp'], float(r['abnormal_prob']), r))

        if logs:
            df_logs = pd.DataFrame(logs)
            # drop repetitive 'action' column if present
            if 'action' in df_logs.columns:
                df_logs = df_logs.drop(columns=['action'])
            # ensure timestamp is datetime and sort
            if 'timestamp' in df_logs.columns:
                df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
                df_logs = df_logs.sort_values('timestamp', ascending=False)
            st.dataframe(df_logs[['timestamp', 'level', 'prob', 'reason']].style.hide_index())
            csv = df_logs[['timestamp', 'level', 'prob', 'reason']].to_csv(index=False)
            st.download_button('Download agent suggestions', csv, file_name='agent_suggestions.csv', mime='text/csv')
        else:
            st.info('No anomalies detected at threshold >= 0.6')


# Ensure Streamlit executes the app when module is imported by `streamlit run`
try:
    main()
except Exception:
    pass

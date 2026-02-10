import streamlit as st
import pandas as pd
import joblib
import time
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
    st.set_page_config(page_title='Smart Factory Anomaly Alert Agent', layout='wide')
    st.title('Smart Factory Anomaly Alert Agent')

    st.sidebar.header('Controls')
    # fuzzy / no_fuzzy mode selector
    mode_options = ['no_fuzzy', 'fuzzy']
    selected_mode = st.sidebar.selectbox('Data mode', mode_options, index=0)
    # dataset is fixed per selected_mode
    if selected_mode == 'no_fuzzy':
        selected_data = 'data/sensor_data.csv'
    else:
        selected_data = 'data/sensor_data_fuzzy.csv'
    
    # model selector (only models for selected mode)
    model_dir = Path('models') / selected_mode
    model_paths = list(model_dir.glob('*.pkl')) if model_dir.exists() else []
    model_names = [p.name for p in model_paths]
    default_idx = 0 if len(model_paths) == 0 else 1
    selected_model_name = st.sidebar.selectbox('Select model', model_names, index=default_idx)
    selected_model_path = None
    selected_model_path = str(next(p for p in model_paths if p.name == selected_model_name))

    # file uploader + run button in Controls
    uploaded = st.sidebar.file_uploader('Upload CSV for analysis', type=['csv'])
    run_analysis = st.sidebar.button('Run analysis')

    df, X, y = load_data(path=selected_data)

    # show per-model metrics for selected mode if available
    metrics_path = f'outputs/{selected_mode}/per_model_metrics.csv'
    if Path(metrics_path).exists():
        try:
            df_metrics = pd.read_csv(metrics_path)
            st.sidebar.markdown('**Per-model metrics**')
            st.sidebar.dataframe(df_metrics, hide_index=True)
        except Exception:
            st.sidebar.info('No metrics available for selected mode')

    # Data Viewer (simple dataset preview)
    df_view = df

    # ensure Data Viewer sorted by timestamp (oldest -> newest) and hide index
    if 'timestamp' in df_view.columns:
        df_view = df_view.copy()
        df_view['timestamp'] = pd.to_datetime(df_view['timestamp'])
        df_view = df_view.sort_values('timestamp', ascending=False).reset_index(drop=True)

    # st.header('Data Viewer')
    # st.dataframe(df_view, hide_index=True)

    # Data load & real-time analysis (no model comparison)
    st.subheader('Data Load & Real-time Analysis')
    st.write('Upload a CSV or select a dataset from the sidebar, then press **Run analysis** to detect anomalies.')
    # uploader and run button are in the Controls sidebar

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
        # ensure spinner visible for at least 0.5s while analysis and dashboard render
        start = time.perf_counter()
        with st.spinner('Running analysis...'):
            # Try to load the selected model (or fallback to heuristic)
            model = None
            if selected_model_path:
                try:
                    model = load_model(selected_model_path)
                    st.write('Using model:', Path(selected_model_path).name)
                except Exception as e:
                    st.error('Failed to load selected model: ' + str(e))
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

            # ensure at least 0.5s spinner duration
            elapsed = time.perf_counter() - start
            remaining = 2 - elapsed
            if remaining > 0:
                time.sleep(remaining)

        # st.subheader('Detected Anomalies (>= 0.6)')
        # # sort by timestamp oldest -> newest and hide index
        anomalies = df2[df2['abnormal_prob'] >= 0.6].copy()
        # if 'timestamp' in anomalies.columns:
        #     anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
        #     anomalies = anomalies.sort_values('timestamp', ascending=True)
        # df_anom_display = anomalies[['timestamp', 'temp', 'pressure', 'vibration', 'abnormal_prob', 'alert_level']].reset_index(drop=True)
        # st.dataframe(df_anom_display)

        # render dashboard using the dataset used for analysis and selected model
        try:
            render_dashboard(df_to_use, X_to_use, y_to_use, selected_model_path=selected_model_path)
        except Exception as e:
            st.error('Failed to render dashboard: ' + str(e))

        st.subheader('AI Agent Suggestions')
        logs = []
        for _, r in anomalies.iterrows():
            logs.append(generate_alert(r['timestamp'], float(r['abnormal_prob']), r))

        if logs:
            df_logs = pd.DataFrame(logs)
            # Remove 'action' column if present and sort by timestamp, hide index
            if 'action' in df_logs.columns:
                df_logs = df_logs.drop(columns=['action'])
            if 'timestamp' in df_logs.columns:
                df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
                df_logs = df_logs.sort_values('timestamp', ascending=False).reset_index(drop=True)
            cols_to_show = [c for c in ['timestamp', 'level', 'prob', 'reason'] if c in df_logs.columns]
            st.dataframe(df_logs[cols_to_show].reset_index(drop=True), hide_index=True)
            csv = df_logs[cols_to_show].to_csv(index=False)
            st.download_button('Download agent suggestions', csv, file_name='agent_suggestions.csv', mime='text/csv')
        else:
            st.info('No anomalies detected at threshold >= 0.6')


# Ensure Streamlit executes the app when module is imported by `streamlit run`
try:
    main()
except Exception:
    pass

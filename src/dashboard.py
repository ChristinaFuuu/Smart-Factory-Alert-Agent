import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
from src.utils import get_probabilities
from src.agent import generate_alert


@st.cache_data
def load_model_cached(path):
    return joblib.load(path)


def compute_fallback_score(df):
    nums = ['temp', 'pressure', 'vibration']
    vals = df[nums].astype(float)
    z = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-9)
    score = z.abs().sum(axis=1) / (3 * 3.0)
    return score.clip(0, 1)


def render_dashboard(df, X, y):
    st.subheader('Dashboard')

    df2 = df.copy()
    if 'timestamp' in df2.columns:
        df2['timestamp'] = pd.to_datetime(df2['timestamp'])
        df2 = df2.sort_values('timestamp')
    else:
        st.warning("No 'timestamp' column found — time charts disabled")

    model_files = list(Path('models').glob('*.pkl'))
    use_model = st.checkbox('Use trained model for scoring', value=bool(model_files))

    if use_model and model_files:
        names = [p.name for p in model_files]
        sel_name = st.selectbox('Select model', names, index=0)
        model_path = str(next(p for p in model_files if p.name == sel_name))
        try:
            model = load_model_cached(model_path)
            probs = get_probabilities(model, X)
            if probs is None:
                df2['abnormal_prob'] = compute_fallback_score(df2)
                st.info('Model loaded but no probabilities available — using heuristic fallback')
            else:
                df2['abnormal_prob'] = probs
        except Exception as e:
            st.error('Failed to load model: ' + str(e))
            df2['abnormal_prob'] = compute_fallback_score(df2)
    else:
        df2['abnormal_prob'] = compute_fallback_score(df2)

    df2['alert_level'] = df2['abnormal_prob'].apply(lambda p: 'CRITICAL' if p >= 0.8 else ('WARNING' if p >= 0.6 else 'NORMAL'))

    # Date range filter
    if 'timestamp' in df2.columns:
        min_d = df2['timestamp'].min().date()
        max_d = df2['timestamp'].max().date()
        dr = st.date_input('Date range', value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if isinstance(dr, tuple) and len(dr) == 2:
            start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_filtered = df2[(df2['timestamp'] >= start) & (df2['timestamp'] <= end)]
        else:
            df_filtered = df2
    else:
        df_filtered = df2

    total = len(df_filtered)
    counts = df_filtered['alert_level'].value_counts().to_dict()
    normal = counts.get('NORMAL', 0)
    warning = counts.get('WARNING', 0)
    critical = counts.get('CRITICAL', 0)

    c1, c2, c3 = st.columns(3)

    def pct(n):
        return f"{(n / total * 100):.1f}%" if total > 0 else "0.0%"

    c1.metric('NORMAL', f"{normal} ({pct(normal)})")
    c2.metric('WARNING', f"{warning} ({pct(warning)})")
    c3.metric('CRITICAL', f"{critical} ({pct(critical)})")

    st.markdown('### Feature time-series')
    features = ['temp', 'pressure', 'vibration']
    if 'timestamp' in df_filtered.columns:
        df_ts = df_filtered.set_index('timestamp')[features]
        if len(df_ts) > 2000:
            df_ts = df_ts.resample('1H').mean()
        st.line_chart(df_ts)
    else:
        st.info('No timestamp available — cannot draw time series')

    st.markdown('### Recent anomalies (>= 0.6)')
    anomalies = df_filtered[df_filtered['abnormal_prob'] >= 0.6].sort_values('abnormal_prob', ascending=False)
    if not anomalies.empty:
        cols = []
        if 'timestamp' in anomalies.columns:
            cols.append('timestamp')
        cols += ['temp', 'pressure', 'vibration', 'abnormal_prob', 'alert_level']
        st.dataframe(anomalies[cols])
        for _, r in anomalies.iterrows():
            label = f"{r['alert_level']} ({r['abnormal_prob']:.2f})"
            with st.expander(f"{r.get('timestamp', '')}: {label}"):
                st.write(generate_alert(r.get('timestamp', ''), float(r['abnormal_prob']), r))
    else:
        st.info('No anomalies in selected range')

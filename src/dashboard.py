import streamlit as st
import pandas as pd
import altair as alt
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

    def pct(n):
        return f"{(n / total * 100):.1f}%" if total > 0 else "0.0%"

    # Colored metric cards using simple HTML
    col1, col2, col3 = st.columns(3)
    card_style = "padding:12px;border-radius:8px;color:white;text-align:center;font-weight:600"
    col1.markdown(f"<div style='{card_style};background:#2ecc71'>NORMAL<br><span style='font-size:20px'>{normal} ({pct(normal)})</span></div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='{card_style};background:#f1c40f;color:#222'>WARNING<br><span style='font-size:20px;color:#222'>{warning} ({pct(warning)})</span></div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='{card_style};background:#e74c3c'>CRITICAL<br><span style='font-size:20px'>{critical} ({pct(critical)})</span></div>", unsafe_allow_html=True)

    st.markdown('### Feature time-series')
    features = ['temp', 'pressure', 'vibration']
    if 'timestamp' in df_filtered.columns:
        df_plot = df_filtered.reset_index().copy()
        # Downsample / resample if too many points
        if len(df_plot) > 2000:
            df_plot = df_plot.set_index('timestamp').resample('1H').mean().reset_index()

        col1, col2, col3 = st.columns(3)

        def chart_for(feature, container):
            with container:
                st.markdown(f'**{feature}**')
                if feature not in df_plot.columns:
                    st.info(f'No `{feature}` data')
                    return
                chart = alt.Chart(df_plot).mark_line().encode(
                    x=alt.X('timestamp:T', title='Time'),
                    y=alt.Y(f'{feature}:Q', title=feature)
                )
                points = alt.Chart(df_plot).mark_circle(size=40).encode(
                    x='timestamp:T',
                    y=alt.Y(f'{feature}:Q'),
                    color=alt.Color('alert_level:N', scale=alt.Scale(domain=['NORMAL','WARNING','CRITICAL'], range=['#2ecc71','#f1c40f','#e74c3c'])),
                    tooltip=['timestamp:T', f'{feature}:Q', 'alert_level:N']
                )
                st.altair_chart((chart + points).interactive(), use_container_width=True)

        chart_for('temp', col1)
        chart_for('pressure', col2)
        chart_for('vibration', col3)
    else:
        st.info('No timestamp available — cannot draw time series')

    st.markdown('### Recent anomalies (>= 0.6)')
    anomalies = df_filtered[df_filtered['abnormal_prob'] >= 0.6].copy()
    if not anomalies.empty:
        if 'timestamp' in anomalies.columns:
            anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
            anomalies = anomalies.sort_values('timestamp', ascending=True)
        cols = []
        if 'timestamp' in anomalies.columns:
            cols.append('timestamp')
        cols += ['temp', 'pressure', 'vibration', 'abnormal_prob', 'alert_level']
        df_anom_display = anomalies[cols].reset_index(drop=True)
        st.dataframe(df_anom_display)
        for _, r in anomalies.iterrows():
            label = f"{r['alert_level']} ({r['abnormal_prob']:.2f})"
            with st.expander(f"{r.get('timestamp', '')}: {label}"):
                st.write(generate_alert(r.get('timestamp', ''), float(r['abnormal_prob']), r))
    else:
        st.info('No anomalies in selected range')

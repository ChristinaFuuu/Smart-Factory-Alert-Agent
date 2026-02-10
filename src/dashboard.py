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


def render_dashboard(df, X, y, selected_model_path=None):
    st.subheader('Dashboard')

    df2 = df.copy()
    if 'timestamp' in df2.columns:
        df2['timestamp'] = pd.to_datetime(df2['timestamp'])
        df2 = df2.sort_values('timestamp')
    else:
        st.warning("No 'timestamp' column found — time charts disabled")

    # Use model selected from sidebar (passed-in `selected_model_path`),
    # otherwise use heuristic fallback scoring.
    if selected_model_path:
        try:
            model = load_model_cached(selected_model_path)
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

        def chart_for(feature):
            st.markdown(f'**{feature}**')
            if feature not in df_plot.columns:
                st.info(f'No `{feature}` data')
                return
            # determine y-domain (min/max) for this feature
            try:
                vals = pd.to_numeric(df_plot[feature], errors='coerce').dropna()
                min_v = float(vals.min()) if not vals.empty else 0.0
                max_v = float(vals.max()) if not vals.empty else 1.0
                # add a small padding so points don't sit on the chart edge
                if min_v == max_v:
                    # expand single-value range
                    min_v -= 0.5
                    max_v += 0.5
                else:
                    span = max_v - min_v
                    pad = span * 0.08  # 8% padding top/bottom
                    min_v -= pad
                    max_v += pad
            except Exception:
                min_v, max_v = 0.0, 1.0

            # compute x-axis domain padding for timestamp so points aren't clipped
            try:
                times = pd.to_datetime(df_plot['timestamp']).dropna()
                min_t = times.min()
                max_t = times.max()
                if min_t == max_t:
                    min_t -= pd.Timedelta(seconds=30)
                    max_t += pd.Timedelta(seconds=30)
                else:
                    span = max_t - min_t
                    pad = pd.Timedelta(seconds=int(span.total_seconds() * 0.08))
                    min_t -= pad
                    max_t += pad
                x_scale = alt.Scale(domain=[min_t.isoformat(), max_t.isoformat()])
                x_enc = alt.X('timestamp:T', title='Time', scale=x_scale)
            except Exception:
                x_enc = alt.X('timestamp:T', title='Time')

            chart = alt.Chart(df_plot).mark_line().encode(
                x=x_enc,
                y=alt.Y(f'{feature}:Q', title=feature, scale=alt.Scale(domain=[min_v, max_v]))
            ).properties(height=220)
            points = alt.Chart(df_plot).mark_circle(size=40).encode(
                x=alt.X('timestamp:T', title='Time', scale=x_scale) if 'x_scale' in locals() else alt.X('timestamp:T', title='Time'),
                y=alt.Y(f'{feature}:Q', scale=alt.Scale(domain=[min_v, max_v])),
                color=alt.Color('alert_level:N', scale=alt.Scale(domain=['NORMAL','WARNING','CRITICAL'], range=['#2ecc71','#f1c40f','#e74c3c'])),
                tooltip=['timestamp:T', f'{feature}:Q', 'alert_level:N']
            ).properties(height=220)
            # render full-width chart stacked in its own row
            st.altair_chart((chart + points).interactive(), width='stretch')

        # render each feature chart in its own row (full width)
        chart_for('temp')
        chart_for('pressure')
        chart_for('vibration')
    else:
        st.info('No timestamp available — cannot draw time series')

    st.markdown('### Recent anomalies (>= 0.6)')
    anomalies = df_filtered[df_filtered['abnormal_prob'] >= 0.6].copy()
    if not anomalies.empty:
        if 'timestamp' in anomalies.columns:
            anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
            anomalies = anomalies.sort_values('timestamp', ascending=False).reset_index(drop=True)
        cols = []
        if 'timestamp' in anomalies.columns:
            cols.append('timestamp')
        cols += ['temp', 'pressure', 'vibration', 'abnormal_prob', 'alert_level']
        df_anom_display = anomalies[cols].reset_index(drop=True)
        st.dataframe(df_anom_display, hide_index=True)
        # for _, r in anomalies.iterrows():
        #     label = f"{r['alert_level']} ({r['abnormal_prob']:.2f})"
        #     with st.expander(f"{r.get('timestamp', '')}: {label}"):
        #         st.write(generate_alert(r.get('timestamp', ''), float(r['abnormal_prob']), r))
    else:
        st.info('No anomalies in selected range')

import argparse
import csv
import random
import time
from collections import deque
from datetime import datetime, timedelta
import math
import os
import joblib
import numpy as np
import pandas as pd


def load_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            def to_float_or_none(val):
                if val is None:
                    return None
                s = str(val).strip()
                if s == '':
                    return None
                try:
                    return float(s)
                except Exception:
                    return None

            rows.append({
                'timestamp': r.get('timestamp') or '',
                'temp': to_float_or_none(r.get('temp') or r.get('temperature')),
                'pressure': to_float_or_none(r.get('pressure')),
                'vibration': to_float_or_none(r.get('vibration')),
            })
    return rows


def compute_stats(samples):
    # compute mean and std for each feature
    # sensible defaults when no valid data present
    defaults = {'temp': (47.5, 1), 'pressure': (1.02, 1), 'vibration': (0.03, 1)}
    if not samples:
        return defaults

    def stats(key):
        vals = [s[key] for s in samples if s.get(key) is not None]
        if not vals:
            return defaults[key]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 1e-6
        return mean, std

    return {k: stats(k) for k in ['temp', 'pressure', 'vibration']}


def compute_medians(samples):
    # compute medians ignoring None, return sensible defaults if missing
    defaults = {'temp': 47.5, 'pressure': 1.02, 'vibration': 0.03}
    if not samples:
        return defaults
    med = {}
    for k in ['temp', 'pressure', 'vibration']:
        vals = [s[k] for s in samples if s.get(k) is not None]
        if not vals:
            med[k] = defaults[k]
        else:
            med[k] = float(np.median(vals))
    return med


def synth_sample(now, stats, anomaly_rate=0.1):
    mean_t, std_t = stats['temp']
    mean_p, std_p = stats['pressure']
    mean_v, std_v = stats['vibration']
    temp = random.gauss(mean_t, std_t)
    pressure = random.gauss(mean_p, std_p)
    vibration = abs(random.gauss(mean_v, std_v))
    label = 'normal'
    if random.random() < anomaly_rate:
        label = 'abnormal'
        # pick one or two features to corrupt
        choice = random.choice(['temp_low','temp_high','press_low','press_high','vib_high'])
        if choice == 'temp_low':
            temp -= abs(random.uniform(5, 15))
        elif choice == 'temp_high':
            temp += abs(random.uniform(5, 15))
        elif choice == 'press_low':
            pressure -= abs(random.uniform(0.2, 0.6))
        elif choice == 'press_high':
            pressure += abs(random.uniform(0.2, 0.6))
        elif choice == 'vib_high':
            vibration += abs(random.uniform(0.02, 0.2))
    return {'timestamp': now.isoformat(timespec='seconds'), 'temp': round(temp, 2), 'pressure': round(pressure, 3), 'vibration': round(vibration, 3), 'label': label}


def detect_window(window, stats, model=None, threshold=0.5):
    # window: list of 3 samples (oldest..newest)
    # We still print the full window externally, but perform detection only on the latest sample.
    latest = window[-1]

    model_score = None
    if model is not None:
        X = pd.DataFrame([{'temp': latest['temp'], 'pressure': latest['pressure'], 'vibration': latest['vibration']}])
        try:
            probs = model.predict_proba(X)
            if probs.shape[1] == 2:
                model_score = float(probs[0][1])
            else:
                model_score = float(probs[0])
        except Exception:
            model_score = None

    # If model available, use its probability; otherwise fallback to 0.0
    score = model_score if model_score is not None else 0.0

    reason = None
    suggestions = []
    primary = None

    # If model indicates anomaly, determine which raw feature(s) deviate from baseline
    if score > threshold:
        abnormal_fields = []
        for k in ['temp', 'pressure', 'vibration']:
            mean, std = stats[k]
            low = mean - 2 * std
            high = mean + 2 * std
            val = latest[k]
            if val < low:
                abnormal_fields.append((k, 'low', val, mean, std))
            elif val > high:
                abnormal_fields.append((k, 'high', val, mean, std))

        if not abnormal_fields:
            # if none exceed 2*std, pick the feature with largest relative deviation
            rel = []
            for k in ['temp', 'pressure', 'vibration']:
                mean, std = stats[k]
                dev = abs(latest[k] - mean) / (std if std > 0 else 1e-6)
                rel.append((dev, k))
            rel.sort(reverse=True)
            k = rel[0][1]
            mean, std = stats[k]
            direction = 'high' if latest[k] > mean else 'low'
            abnormal_fields.append((k, direction, latest[k], mean, std))

        # build human-readable reason and suggestions
        parts = []
        for idx, (k, dirc, val, mean, std) in enumerate(abnormal_fields):
            if k == 'temp':
                if dirc == 'low':
                    parts.append('Temperature below expected')
                    suggestions.append('- Temperature below expected — check heater or process settings.')
                else:
                    parts.append('Temperature above expected')
                    suggestions.append('- Elevated temperature — inspect cooling system and heat sources.')
            elif k == 'pressure':
                if dirc == 'low':
                    parts.append('Pressure below expected')
                    suggestions.append('- Pressure lower than normal — verify pump and supply lines.')
                else:
                    parts.append('Pressure above expected')
                    suggestions.append('- Pressure higher than normal — check relief valves and possible blockages.')
            elif k == 'vibration':
                parts.append('Vibration higher than normal')
                suggestions.append('- Increased vibration — inspect bearings, couplings, and motor alignment.')
            if primary is None:
                primary = k

        reason = '; '.join(parts)
    else:
        reason = 'Within expected range'

    return score, reason, suggestions, primary


def print_window(window):
    print('Preprocess the data..')
    print('    timestamp          temp    pressure  vibration')
    for i, s in enumerate(window):
        ts = s.get('timestamp', '')
        t = s.get('temp')
        p = s.get('pressure')
        v = s.get('vibration')
        temp_str = f"{t:7.2f}" if t is not None else '   --  '
        pres_str = f"{p:9.3f}" if p is not None else '   --    '
        vib_str = f"{v:10.3f}" if v is not None else '   --     '
        print(f"{ts:19s} {temp_str} {pres_str} {vib_str}")
    print('-' * 64)


def main():
    parser = argparse.ArgumentParser(description='CLI sensor data simulator and anomaly detector')
    parser.add_argument('--rate', type=float, default=1.0, help='seconds between samples')
    parser.add_argument('--count', type=int, default=0, help='number of samples to emit (0 for infinite)')
    parser.add_argument('--anomaly-rate', type=float, default=0.2, help='probability a sample is anomalous')
    parser.add_argument('--source', choices=['synth','csv'], default='csv', help='data source')
    parser.add_argument('--csv-path', default='data/simulate/sensor_data.csv', help='path to sensor csv')
    parser.add_argument('--model-path', default='models/no_fuzzy/random_forest.pkl', help='path to saved joblib model (optional)')
    parser.add_argument('--threshold', type=float, default=0.6, help='probability threshold for anomaly when using model')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    csv_rows = []
    if args.source == 'csv':
        csv_rows = load_csv(args.csv_path)

    # compute medians for imputation when CSV has missing values
    medians = compute_medians(csv_rows) if csv_rows else compute_medians(None)

    # compute baseline stats from CSV if available, else defaults
    stats = compute_stats(csv_rows if csv_rows else None)

    # load model if available
    model = None
    if args.model_path:
        try:
            if os.path.exists(args.model_path):
                model = joblib.load(args.model_path)
            else:
                # try relative to models folder
                p = os.path.join('models', args.model_path)
                if os.path.exists(p):
                    model = joblib.load(p)
        except Exception:
            model = None

    window = deque(maxlen=3)
    now = datetime.now()
    emitted = 0
    csv_idx = 0
    try:
        while True:
            if args.source == 'csv' and csv_rows:
                base = csv_rows[csv_idx % len(csv_rows)]
                # impute missing values using medians computed from CSV
                t = base.get('temp') if base.get('temp') is not None else medians['temp']
                p = base.get('pressure') if base.get('pressure') is not None else medians['pressure']
                v = base.get('vibration') if base.get('vibration') is not None else medians['vibration']
                sample = {'timestamp': (now + timedelta(seconds=emitted)).isoformat(timespec='seconds'), 'temp': round(t, 2), 'pressure': round(p, 3), 'vibration': round(v, 3)}
                # optionally inject anomaly
                # if random.random() < args.anomaly_rate:
                #     sample['label'] = 'abnormal'
                #     # simple injection
                #     sample['temp'] += random.choice([-8,8])
                csv_idx += 1
            else:
                sample = synth_sample(now + timedelta(seconds=emitted), stats, anomaly_rate=args.anomaly_rate)

            window.append(sample)

            print('\nReceive sensor data..')
            print_window(list(window))

            # Run detection whenever we have at least one sample (so first two samples are classified)
            if len(window) >= 1:
                score, reason, suggestions, primary = detect_window(list(window), stats, model=model, threshold=args.threshold)
                label = 'abnormal' if score > args.threshold else 'normal'
                display_label = f"{label}" if label == 'normal' else label
                print(f"\nPrediction result: {display_label} (based on {len(window)} sample(s))")
                print(f'Confidence: {score:.2f}')
                ts = window[-1].get('timestamp', '')
                if label == 'abnormal':
                    print(f"\n{ts} - [ABNORMAL!] anomaly_score: {score:.3f}, temp: {window[-1].get('temp')}, pressure: {window[-1].get('pressure')}, vibration: {window[-1].get('vibration')}")
                    print('⚠️ ALERT! Anomaly on machine detected.')
                    for s in suggestions:
                        print(s)
                else:
                    print(f"\n{ts} - 🟢 OK anomaly_score: {score:.3f}")

            emitted += 1
            if args.count and emitted >= args.count:
                break
            time.sleep(max(0.0, args.rate))
    except KeyboardInterrupt:
        print('\nSimulation interrupted by user')


if __name__ == '__main__':
    main()

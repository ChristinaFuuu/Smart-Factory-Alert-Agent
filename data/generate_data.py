import csv
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_sensor_data(n=400, start=None, out_path='data/sensor_data.csv'):
    if start is None:
        start = datetime.now() - timedelta(minutes=n)

    timestamps = [start + timedelta(minutes=i) for i in range(n)]

    # Base normal distributions
    temp = np.random.normal(loc=47.5, scale=0.8, size=n)
    pressure = np.random.normal(loc=1.02, scale=0.01, size=n)
    vibration = np.random.normal(loc=0.03, scale=0.005, size=n)

    # Inject abnormal samples (~18%)
    num_abnormal = int(n * 0.18)
    abnormal_idx = set(random.sample(range(n), num_abnormal))

    for i in abnormal_idx:
        choice = random.choice(['temp_hi', 'temp_lo', 'pressure_hi', 'pressure_lo', 'vib_hi'])
        if choice == 'temp_hi':
            temp[i] = np.random.uniform(52.5, 60.0)
        elif choice == 'temp_lo':
            temp[i] = np.random.uniform(35.0, 42.5)
        elif choice == 'pressure_hi':
            pressure[i] = np.random.uniform(1.09, 1.15)
        elif choice == 'pressure_lo':
            pressure[i] = np.random.uniform(0.90, 0.96)
        elif choice == 'vib_hi':
            vibration[i] = np.random.uniform(0.08, 0.15)

    # Add slight random noise for realism
    temp += np.random.normal(scale=0.05, size=n)
    pressure += np.random.normal(scale=0.002, size=n)
    vibration += np.random.normal(scale=0.002, size=n)

    rows = []
    for i in range(n):
        label = 'abnormal' if i in abnormal_idx else 'normal'
        rows.append({
            'timestamp': timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
            'temp': round(float(temp[i]), 3),
            'pressure': round(float(pressure[i]), 4),
            'vibration': round(float(vibration[i]), 4),
            'label': label,
        })

    df = pd.DataFrame(rows)

    # Inject missing values ~2.5%
    total_cells = df.shape[0] * 3  # only numerical columns
    num_missing = max(1, int(total_cells * 0.025))
    for _ in range(num_missing):
        r = random.randrange(n)
        c = random.choice(['temp', 'pressure', 'vibration'])
        df.at[r, c] = np.nan

    # Ensure output folder exists
    df.to_csv(out_path, index=False)
    print(f'Saved {len(df)} rows to {out_path}')


if __name__ == '__main__':
    generate_sensor_data(n=420)

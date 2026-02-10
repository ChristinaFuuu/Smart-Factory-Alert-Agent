import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import argparse

def generate_sensor_data_compatible(n=400, start=None, out_path='data/sensor_data.csv',
                                    enable_fuzzy=True,
                                    fuzzy_prob_temp_low=0.3,
                                    fuzzy_prob_temp_high=0.3,
                                    fuzzy_prob_pressure_low=0.3,
                                    fuzzy_prob_pressure_high=0.3,
                                    fuzzy_abnormal_ratio=0.2,
                                    random_seed=42):
    # 固定隨機種子，確保可重現
    random.seed(random_seed)
    np.random.seed(random_seed)

    if start is None:
        start = datetime.now() - timedelta(minutes=n)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]

    # 閾值區間定義
    temp_normal_range = (45, 50)
    temp_low_fuzzy_range = (43, 45)
    temp_high_fuzzy_range = (50, 52)

    pressure_normal_range = (1.00, 1.05)
    pressure_low_fuzzy_range = (0.97, 1.00)
    pressure_high_fuzzy_range = (1.05, 1.08)

    vib_threshold = 0.07

    # 生成容器
    temp = np.zeros(n)
    pressure = np.zeros(n)
    vibration = np.zeros(n)

    # 產生連續性較好的時間序列：周期分量 + AR(1) 隨機漫步 + 小幅平滑
    def synthesize_continuous_signal(mean, n, rho=0.95, noise_scale=0.1, amp_range=(0.1, 1.0), cycles_range=(0.5, 3.0)):
        cycles = np.random.uniform(*cycles_range)
        period = max(3.0, n / cycles)
        amp = np.random.uniform(*amp_range)
        phase = np.random.uniform(0, 2 * np.pi)
        idx = np.arange(n)
        baseline = mean + amp * np.sin(2 * np.pi * idx / period + phase)

        arr = np.zeros(n)
        arr[0] = baseline[0] + np.random.normal(scale=noise_scale)
        for i in range(1, n):
            arr[i] = rho * arr[i-1] + (1 - rho) * baseline[i] + np.random.normal(scale=noise_scale)

        # 簡單 EMA 平滑，減少高頻抖動
        alpha = 0.2
        for i in range(1, n):
            arr[i] = alpha * arr[i] + (1 - alpha) * arr[i-1]
        return arr

    if not enable_fuzzy:
        # 使用連續訊號產生較平滑的時間序列
        temp = synthesize_continuous_signal(47.5, n, rho=0.95, noise_scale=0.08,
                                            amp_range=(0.2, 1.0), cycles_range=(0.5, 2.5))
        pressure = synthesize_continuous_signal(1.02, n, rho=0.97, noise_scale=0.005,
                                                amp_range=(0.002, 0.02), cycles_range=(0.5, 3.0))
        vibration = synthesize_continuous_signal(0.03, n, rho=0.8, noise_scale=0.003,
                                                 amp_range=(0.001, 0.02), cycles_range=(2.0, 8.0))

        # 產生連續區段的異常（較真實：短時間持續異常）
        num_abnormal = int(n * 0.18)
        abnormal_idx = set()
        max_seg = max(1, int(n * 0.03))
        while len(abnormal_idx) < num_abnormal:
            start = random.randrange(n)
            seg_len = random.randint(1, max_seg)
            for j in range(start, min(n, start + seg_len)):
                abnormal_idx.add(j)

        for i in list(abnormal_idx):
            choice = random.choice(['temp_hi', 'temp_lo', 'pressure_hi', 'pressure_lo', 'vib_hi'])
            if choice == 'temp_hi':
                temp[i] += np.random.uniform(5.0, 12.0)
            elif choice == 'temp_lo':
                temp[i] -= np.random.uniform(5.0, 12.0)
            elif choice == 'pressure_hi':
                pressure[i] += np.random.uniform(0.06, 0.13)
            elif choice == 'pressure_lo':
                pressure[i] -= np.random.uniform(0.08, 0.12)
            elif choice == 'vib_hi':
                vibration[i] += np.random.uniform(0.05, 0.12)

        # 小幅微調雜訊
        temp += np.random.normal(scale=0.02, size=n)
        pressure += np.random.normal(scale=0.001, size=n)
        vibration += np.random.normal(scale=0.001, size=n)

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

        # 注入缺失值 2.5%
        total_cells = df.shape[0] * 3
        num_missing = max(1, int(total_cells * 0.025))
        for _ in range(num_missing):
            r = random.randrange(n)
            c = random.choice(['temp', 'pressure', 'vibration'])
            df.at[r, c] = np.nan

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f'[No Fuzzy] Saved {len(df)} rows to {out_path}')
        return

    # --- 模糊區間啟用版 ---
    # 策略：先產生明確異常資料，然後只將其中部分（fuzzy_abnormal_ratio）調整到模糊區間
    
    # 先用連續訊號生成初始數據（較接近真實感測序列）
    temp = synthesize_continuous_signal(47.5, n, rho=0.95, noise_scale=0.08,
                                        amp_range=(0.2, 1.0), cycles_range=(0.5, 2.5))
    pressure = synthesize_continuous_signal(1.02, n, rho=0.97, noise_scale=0.005,
                                            amp_range=(0.002, 0.02), cycles_range=(0.5, 3.0))
    vibration = synthesize_continuous_signal(0.03, n, rho=0.8, noise_scale=0.003,
                                             amp_range=(0.001, 0.02), cycles_range=(2.0, 8.0))

    # 產生連續區段的異常（較真實：短時間持續異常）
    num_abnormal = int(n * 0.18)
    abnormal_idx = set()
    max_seg = max(1, int(n * 0.03))
    
    while len(abnormal_idx) < num_abnormal:
        start_idx = random.randrange(n)
        seg_len = random.randint(1, max_seg)
        for j in range(start_idx, min(n, start_idx + seg_len)):
            abnormal_idx.add(j)

    # 先將所有異常資料設為明確異常（超出閾值）
    for i in list(abnormal_idx):
        choice = random.choice(['temp_hi', 'temp_lo', 'pressure_hi', 'pressure_lo', 'vib_hi'])
        if choice == 'temp_hi':
            temp[i] += np.random.uniform(5.0, 12.0)
        elif choice == 'temp_lo':
            temp[i] -= np.random.uniform(5.0, 12.0)
        elif choice == 'pressure_hi':
            pressure[i] += np.random.uniform(0.06, 0.13)
        elif choice == 'pressure_lo':
            pressure[i] -= np.random.uniform(0.08, 0.12)
        elif choice == 'vib_hi':
            vibration[i] += np.random.uniform(0.05, 0.12)

    # 現在選擇部分異常資料（fuzzy_abnormal_ratio，例如 20%）調整到模糊區間
    # 這樣會有：80% 明確異常（容易偵測） + 20% 模糊區間異常（較難偵測）
    num_fuzzy_abnormal = int(len(abnormal_idx) * fuzzy_abnormal_ratio)
    fuzzy_count = 0  # 追蹤實際調整的數量
    
    if num_fuzzy_abnormal > 0:
        abnormal_list = list(abnormal_idx)
        fuzzy_candidates = random.sample(abnormal_list, num_fuzzy_abnormal)
        fuzzy_count = len(fuzzy_candidates)
        
        print(f'  調整 {fuzzy_count}/{len(abnormal_idx)} ({fuzzy_abnormal_ratio*100:.0f}%) 個異常資料到模糊區間')
        
        for i in fuzzy_candidates:
            # 將這個異常點調整到模糊區間
            choice = random.choice(['temp', 'pressure', 'both'])
            
            if choice in ['temp', 'both']:
                # 調整溫度到模糊區間
                if random.random() < 0.5:
                    temp[i] = np.random.uniform(*temp_low_fuzzy_range)
                else:
                    temp[i] = np.random.uniform(*temp_high_fuzzy_range)
            
            if choice in ['pressure', 'both']:
                # 調整壓力到模糊區間
                if random.random() < 0.5:
                    pressure[i] = np.random.uniform(*pressure_low_fuzzy_range)
                else:
                    pressure[i] = np.random.uniform(*pressure_high_fuzzy_range)
            
            # 震動設為正常範圍
            vibration[i] = np.random.uniform(0.01, vib_threshold * 0.8)

    # 小幅微調雜訊
    temp += np.random.normal(scale=0.02, size=n)
    pressure += np.random.normal(scale=0.001, size=n)
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

    # 注入缺失值 2.5%
    total_cells = df.shape[0] * 3
    num_missing = max(1, int(total_cells * 0.025))
    for _ in range(num_missing):
        r = random.randrange(n)
        c = random.choice(['temp', 'pressure', 'vibration'])
        df.at[r, c] = np.nan

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    
    clear_count = len(abnormal_idx) - fuzzy_count
    print(f'[Fuzzy Enabled] Saved {len(df)} rows to {out_path}')
    print(f'  總異常: {len(abnormal_idx)} ({len(abnormal_idx)/n*100:.1f}%)')
    print(f'  明確異常: {clear_count} ({clear_count/n*100:.1f}%)')
    print(f'  模糊區間異常: {fuzzy_count} ({fuzzy_count/n*100:.1f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sensor data (no_fuzzy / fuzzy)')
    parser.add_argument('--modes', nargs='+', choices=['no_fuzzy', 'fuzzy'], default=['fuzzy'], help='Which modes to generate')
    parser.add_argument('-n', type=int, default=420, help='Number of rows')
    parser.add_argument('--out_dir', type=str, default='data', help='Output directory for generated CSVs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fuzzy_prob_temp_low', type=float, default=0.5)
    parser.add_argument('--fuzzy_prob_temp_high', type=float, default=0.5)
    parser.add_argument('--fuzzy_prob_pressure_low', type=float, default=0.5)
    parser.add_argument('--fuzzy_prob_pressure_high', type=float, default=0.5)
    parser.add_argument('--fuzzy_abnormal_ratio', type=float, default=0.2, 
                        help='Ratio of abnormal data to move to fuzzy zone (0.0-1.0, default 0.2 means 20%%)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for m in args.modes:
        out_path = os.path.join(args.out_dir, 'sensor_data.csv' if m == 'no_fuzzy' else 'sensor_data_fuzzy.csv')
        enable_fuzzy = (m == 'fuzzy')
        generate_sensor_data_compatible(n=args.n,
                                        enable_fuzzy=enable_fuzzy,
                                        out_path=out_path,
                                        fuzzy_prob_temp_low=args.fuzzy_prob_temp_low,
                                        fuzzy_prob_temp_high=args.fuzzy_prob_temp_high,
                                        fuzzy_prob_pressure_low=args.fuzzy_prob_pressure_low,
                                        fuzzy_prob_pressure_high=args.fuzzy_prob_pressure_high,
                                        fuzzy_abnormal_ratio=args.fuzzy_abnormal_ratio,
                                        random_seed=args.seed)
        print(f'Wrote {out_path}')

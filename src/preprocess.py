import pandas as pd
from sklearn.impute import SimpleImputer


def load_and_preprocess(path='data/sensor_data.csv'):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Impute missing numeric values with median
    imputer = SimpleImputer(strategy='median')
    nums = ['temp', 'pressure', 'vibration']
    df[nums] = imputer.fit_transform(df[nums])

    # Encode label: normal -> 0, abnormal -> 1
    df['label_enc'] = df['label'].map({'normal': 0, 'abnormal': 1})

    X = df[['temp', 'pressure', 'vibration']]
    y = df['label_enc']

    return df, X, y


if __name__ == '__main__':
    df, X, y = load_and_preprocess()
    print('Loaded', len(df), 'rows')

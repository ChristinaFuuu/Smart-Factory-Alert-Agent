import numpy as np


def add_noise(X, noise_level, seed=None, cols=None):
    """Return a copy of X with Gaussian noise added to numeric columns.

    - X: pandas DataFrame or 2D numpy array. If DataFrame, column names are used.
    - noise_level: float multiplier of each column's std (sigma = noise_level * std_col).
    - seed: optional int for RNG reproducibility.
    - cols: list of column names to perturb (defaults to ['temp','pressure','vibration']).

    Returns the same type as X (DataFrame or ndarray).
    """
    try:
        import pandas as _pd
    except Exception:
        _pd = None

    rng = np.random.default_rng(seed)

    if cols is None:
        cols = ['temp', 'pressure', 'vibration']

    # If input is a DataFrame, operate on a copy and preserve dtypes/cols
    if _pd is not None and isinstance(X, _pd.DataFrame):
        Xc = X.copy()
        for c in cols:
            if c in Xc.columns:
                col_vals = Xc[c].astype(float).values
                std = np.std(col_vals, ddof=0)
                sigma = float(noise_level) * std
                if sigma > 0:
                    noise = rng.normal(0, sigma, size=col_vals.shape)
                    Xc[c] = col_vals + noise
        return Xc

    # Otherwise try to handle numpy arrays
    arr = np.asarray(X)
    if arr.ndim != 2:
        return X

    # If cols provided as indices (ints), use them; otherwise do nothing
    idxs = []
    for c in cols:
        if isinstance(c, int) and 0 <= c < arr.shape[1]:
            idxs.append(c)

    if not idxs:
        return arr

    out = arr.copy().astype(float)
    for i in idxs:
        col_vals = out[:, i]
        std = np.std(col_vals, ddof=0)
        sigma = float(noise_level) * std
        if sigma > 0:
            out[:, i] = col_vals + rng.normal(0, sigma, size=col_vals.shape)

    return out

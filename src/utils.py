from sklearn.calibration import CalibratedClassifierCV
import numpy as np


def get_probabilities(model, X):
    """Attempt to get probability estimates for model on X.

    Returns numpy array of probabilities for the positive class, or None if not available.
    """
    # Prefer predict_proba
    try:
        probs = model.predict_proba(X)
        return np.asarray(probs)[:, 1]
    except Exception:
        pass

    # Try decision_function then map to probabilities via calibration
    try:
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
            # Use a calibrated wrapper to convert to probabilities
            try:
                calib = CalibratedClassifierCV(base_estimator=model, cv='prefit')
                calib.fit(X, model.predict(X))
                return calib.predict_proba(X)[:, 1]
            except Exception:
                # fallback: min-max scale scores to [0,1]
                s = np.asarray(scores)
                denom = (s.max() - s.min())
                if denom == 0:
                    return None
                return (s - s.min()) / denom
    except Exception:
        pass

    return None

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=42):
    """
    Load the California Housing dataset and split into train and test sets.
    Returns X_train, X_test, y_train, y_test.
    """
    data = fetch_california_housing()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_model(model, path="model.joblib"):
    """Save a trained model to disk."""
    joblib.dump(model, path)

def load_model(path="model.joblib"):
    """Load a model from disk."""
    return joblib.load(path)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test set and return R^2 and MSE."""
    from sklearn.metrics import r2_score, mean_squared_error
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mse

def quantize_params(coefs, intercept):
    """
    Quantize coefficients and intercept to unsigned 8-bit integers.
    Returns (quant_coefs, quant_intercept, min_val, scale) needed for dequantization.
    """
    # Flatten coefs to 1D array for uniform processing (if not already).
    coefs = np.array(coefs, dtype=float).ravel()
    intercept = float(intercept)
    # Determine scaling based on min and max values across all parameters.
    min_val = min(coefs.min(), intercept)
    max_val = max(coefs.max(), intercept)
    range_val = max_val - min_val
    if range_val == 0:
        # Avoid division by zero (if all values equal, which is unlikely in linear regression).
        scale = 1.0
    else:
        scale = range_val / 255.0
    # Quantize coefficients and intercept.
    quant_coefs = np.round((coefs - min_val) / scale).astype(np.uint8) if range_val != 0 else np.zeros_like(coefs, dtype=np.uint8)
    quant_intercept = int(round((intercept - min_val) / scale)) if range_val != 0 else 0
    # Clip quantized intercept to [0, 255] and cast to uint8.
    quant_intercept = np.uint8(np.clip(quant_intercept, 0, 255))
    return quant_coefs, quant_intercept, min_val, scale

def dequantize_params(quant_coefs, quant_intercept, min_val, scale):
    """
    Reconstruct float coefficients and intercept from their quantized values.
    """
    # Ensure numpy types for calculation
    quant_coefs = np.array(quant_coefs, dtype=float)
    quant_intercept = float(quant_intercept)
    # Dequantize by reversing the scaling and offset
    coefs = quant_coefs * scale + min_val
    intercept = quant_intercept * scale + min_val
    return coefs, intercept

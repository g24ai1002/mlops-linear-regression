import importlib
import os
import sys
import numpy as np

# Ensure the src directory is in PYTHONPATH for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

import pytest
from utils import load_data

def test_load_data():
    """Test that the dataset loads correctly and has expected shape."""
    X_train, X_test, y_train, y_test = load_data()
    # California housing has 8 features and non-empty target
    assert X_train.shape[1] == 8, "Dataset should have 8 features"
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Train/test sets should not be empty"
    assert X_train.shape[0] + X_test.shape[0] == len(y_train) + len(y_test), "Total samples mismatch"
    # Check that the data types are numeric
    assert np.issubdtype(X_train.dtype, np.number), "Features should be numeric"
    assert np.issubdtype(np.array(y_train).dtype, np.number), "Targets should be numeric"

def test_train_model_and_performance():
    """Test that training produces a LinearRegression model with acceptable performance."""
    # Import train module and call train_model function
    train_module = importlib.import_module('train')
    # Ensure the function exists
    assert hasattr(train_module, 'train_model'), "train.py must have a train_model function"
    model, r2, mse = train_module.train_model()
    # Validate model is a LinearRegression instance
    from sklearn.linear_model import LinearRegression
    assert isinstance(model, LinearRegression), "Model should be an instance of LinearRegression"
    # Model should have coefficients (i.e., be trained)
    assert hasattr(model, "coef_"), "Trained model should have attribute coef_"
    assert model.coef_ is not None, "Model coefficients should not be None"
    # Performance should exceed a minimum R^2 threshold (e.g., 0.5)
    assert r2 > 0.5, f"R^2 score {r2} is too low"
    # The model should have saved a model file after training when run as script
    # (Simulate script run to produce model.joblib)
    if os.path.exists("model.joblib"):
        os.remove("model.joblib")
    os.system("python src/train.py")
    assert os.path.exists("model.joblib"), "Model file was not saved by train.py"

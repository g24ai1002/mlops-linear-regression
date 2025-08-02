import numpy as np
from utils import load_model, load_data

if __name__ == "__main__":
    # Load the trained model
    model = load_model("model.joblib")
    # Load dataset and split (same random state to get same test set)
    _, X_test, _, y_test = load_data()
    # Perform predictions on the test set
    y_pred = model.predict(X_test)
    # Print a few sample predictions vs actual values for verification
    print("Sample predictions vs actual:")
    for i in range(5):
        print(f"Predicted: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f}")

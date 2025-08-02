import numpy as np
from sklearn.linear_model import LinearRegression
from utils import load_data, save_model, evaluate_model

def train_model():
    """Train a LinearRegression model on the California housing dataset and return the model and performance metrics."""
    # Load the dataset
    X_train, X_test, y_train, y_test = load_data()
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Evaluate performance on test set
    r2, mse = evaluate_model(model, X_test, y_test)
    return model, r2, mse

if __name__ == "__main__":
    model, r2, mse = train_model()
    # Print performance metrics
    print(f"Test R2 score: {r2:.4f}")
    print(f"Test MSE: {mse:.4f}")
    # Save the trained model to disk
    save_model(model, "model.joblib")

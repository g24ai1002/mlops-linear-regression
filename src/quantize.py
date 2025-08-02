import numpy as np
import joblib
from utils import load_model, save_model, quantize_params, dequantize_params, load_data

if __name__ == "__main__":
    # Load the trained model from disk
    model = load_model("model.joblib")
    # Extract model parameters
    coefs = model.coef_
    intercept = model.intercept_
    # Save raw (float) parameters
    joblib.dump((coefs, intercept), "unquant_params.joblib")
    # Quantize parameters to 8-bit unsigned ints
    quant_coefs, quant_intercept, min_val, scale = quantize_params(coefs, intercept)
    # Save quantized parameters and quantization info
    joblib.dump((quant_coefs, quant_intercept, min_val, scale), "quant_params.joblib")
    # Perform inference with de-quantized weights to evaluate quantization impact
    # Reload test data (same split as training by using identical random_state)
    _, X_test, _, y_test = load_data()
    # Dequantize parameters back to float
    dequant_coefs, dequant_intercept = dequantize_params(quant_coefs, quant_intercept, min_val, scale)
    # Compute predictions using the de-quantized parameters
    y_pred_quant = X_test.dot(dequant_coefs) + dequant_intercept
    # Calculate R^2 and MSE for quantized model
    from sklearn.metrics import r2_score, mean_squared_error
    r2_quant = r2_score(y_test, y_pred_quant)
    mse_quant = mean_squared_error(y_test, y_pred_quant)
    # Also compute original model's performance on the same test set for comparison
    y_pred_orig = model.predict(X_test)
    r2_orig = r2_score(y_test, y_pred_orig)
    mse_orig = mean_squared_error(y_test, y_pred_orig)
    # Print comparison of performance
    print(f"Original Test R2: {r2_orig:.4f}, Original Test MSE: {mse_orig:.4f}")
    print(f"Quantized Test R2: {r2_quant:.4f}, Quantized Test MSE: {mse_quant:.4f}")

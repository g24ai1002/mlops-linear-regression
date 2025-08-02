<img width="1091" height="168" alt="image" src="https://github.com/user-attachments/assets/58f6abc1-8b5f-4e70-833b-3d558098395b" /># MLOps Linear Regression Pipeline

**Name**: Shubham Verma  
**Roll Number**: G24AI1002  
**Course**: MLOps  
**Instructor**: Dr. Pratik Mazumder

---

## 📌 Project Overview

This repository implements a complete **MLOps pipeline** using `LinearRegression` from `scikit-learn` on the **California Housing dataset**. It includes:

- Model training
- Manual quantization (8-bit)
- Inference and evaluation
- Docker containerization
- CI/CD integration with GitHub Actions
- Automated testing

---

## 📂 Repository Structure

```
.
├── src/
│   ├── train.py           # Train Linear Regression model
│   ├── quantize.py        # Quantize model coefficients
│   ├── predict.py         # Make predictions
│   └── utils.py           # Shared functions
├── tests/
│   └── test_train.py      # Unit tests using pytest
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions workflow
├── model.joblib           # Pretrained model (generated via train.py)
├── requirements.txt       # Project dependencies
├── Dockerfile             # Container setup
├── .gitignore             # Ignore cache/artifact files
└── README.md              # Project documentation
```

---

## 🚀 How to Run

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate        # On Windows
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train.py
```

### 3. Quantize the Model
```bash
python src/quantize.py
```

### 4. Predict
```bash
python src/predict.py
```

### 5. Run Unit Tests
```bash
pytest
```

### 6. Build & Run with Docker
```bash
docker build -t housing_model .
docker run --rm housing_model
```

---

## 🧪 CI/CD Workflow

GitHub Actions runs on **every push to `main`**, consisting of:

1. ✅ `test suite`: runs unit tests using `pytest`
2. ✅ `train and quantize`: trains model and performs quantization
3. ✅ `build and test container`: builds Docker image and verifies predictions

All jobs must pass for the workflow to succeed.


---

## 📷 Output

Present as png file in the repo

* Sample Vs Actual Prediction
* R2 Score & MSE

---

## 🔐 Notes & Constraints

- Only `LinearRegression` is used.
- No hardcoded values.
- Code is modularized and organized.
- Only the `main` branch exists.
- CI/CD is implemented using GitHub Actions.

---

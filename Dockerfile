FROM python:3.10-slim
WORKDIR /app
# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
# Copy source code and model artifact into the container
COPY src/*.py .
COPY model.joblib .
# Set default command to run prediction for verification
CMD ["python", "predict.py"]
# Stage 1: Builder (not strictly needed for inference, but good practice)
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Stage 2: Inference server for SageMaker
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy inference code and the trained model artifact.
# This assumes the 'model' directory is present during the build.
COPY --from=builder /app/model/ ./model/
COPY --from=builder /app/inference/ ./inference/
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "inference.predict:app"]
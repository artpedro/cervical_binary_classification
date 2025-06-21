# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Your code
COPY . .

# Standard runtime locations
ENV DATA_DIR=/workspace/data \
    METRICS_DIR=/workspace/metrics


ENTRYPOINT ["python", "train.py"]

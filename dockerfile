# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

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

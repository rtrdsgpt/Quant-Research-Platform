# Quant Research Platform
#
# Build:  docker build -t quant-research-platform .
# Run pipeline:  docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
#                  quant-research-platform --full
# Run API:       docker run --rm -p 8000:8000 quant-research-platform api
#
# Note: this image includes torch/transformers (FinBERT sentiment) and
# lightgbm/xgboost, so the build is large (~3-4 GB) -- that's inherent to
# the model stack, not accidental bloat.

FROM python:3.11-slim AS base

# libgomp1: OpenMP runtime required by lightgbm/xgboost at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    # CPU-only torch build -- the default PyPI wheel drags in several GB
    # of CUDA/cuDNN libraries that are dead weight on a CPU-only image
    # (FinBERT sentiment scoring here never touches a GPU).
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Signals to run.sh that dependencies are already installed system-wide,
# so it should exec straight into main.py / uvicorn instead of managing a venv.
ENV QRP_IN_CONTAINER=1
ENV PYTHONUNBUFFERED=1

RUN mkdir -p data/raw data/processed data/features models reports logs

ENTRYPOINT ["./run.sh"]
CMD ["--full"]

FROM python:3.11-slim

WORKDIR /app

# libgl1 + libglib2.0-0 are the runtime libs OpenCV links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# The default PyPI torch wheel bundles CUDA libraries (~2 GB) that a CPU-only
# Hugging Face Space can never use. The CPU index cuts the image by roughly an
# order of magnitude and speeds up cold starts.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

# Serve one model, chosen by env var so the image does not need rebuilding to
# swap checkpoints.
ENV DFU_MODEL_PATH=/app/model/dfu_resnet18_grouped.pt \
    TORCH_THREADS=2 \
    PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health', timeout=8).status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

# CPU-only image (Cloud Run / local CPU). Avoids multi-GB CUDA base images.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sentiment-analysis-api

COPY requirements.txt .

# CPU torch first (PyPI default Linux wheels are CUDA), then the rest without re-resolving torch.
RUN pip install --no-cache-dir torch==2.6.0 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^torch==' requirements.txt > /tmp/requirements.notorch.txt \
    && pip install --no-cache-dir -r /tmp/requirements.notorch.txt \
    && rm /tmp/requirements.notorch.txt

COPY . .
RUN python scripts/prefetch_models.py

EXPOSE 8001
CMD ["python3", "-u", "-m", "run"]

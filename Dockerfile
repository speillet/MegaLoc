# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/megaloc/.cache/huggingface \
    MEGALOC_DEVICE=auto

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 megaloc \
    && mkdir -p /home/megaloc/.cache/huggingface \
    && chown -R megaloc:megaloc /home/megaloc/.cache

COPY requirements-api.txt ./
RUN python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install -r requirements-api.txt

COPY api.py inference.py megaloc_model.py test_megaloc.py ./

USER megaloc
EXPOSE 8000

# The initial weight download may take several minutes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/megaloc/.cache/huggingface \
    MEGALOC_DEVICE=auto

WORKDIR /app

# Docker supplies predefined proxy variables through --build-arg. Normalize
# their case for APT and pip, and reuse the HTTP proxy for HTTPS if needed.
# Keep this scoped to RUN so proxy settings are not persisted with ENV.
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
           https_proxy="${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}" \
           no_proxy="${no_proxy:-${NO_PROXY:-}}" \
    && sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 -o APT::Update::Error-Mode=any update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 megaloc \
    && mkdir -p /home/megaloc/.cache/huggingface \
    && chown -R megaloc:megaloc /home/megaloc/.cache

COPY requirements.txt ./
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
           https_proxy="${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}" \
           no_proxy="${no_proxy:-${NO_PROXY:-}}" \
    && python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install -r requirements.txt

COPY api.py inference.py megaloc_model.py cli_pipeline.py ./

USER megaloc
EXPOSE 8000

# The initial weight download may take several minutes. Check the local API
# directly even when the container needs a proxy for external downloads.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

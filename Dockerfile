# DeepWeather AI (SIH26078) - backend image (FastAPI + scientific Python stack).
#
#   docker build -t deepweather-ai-backend .
#   docker run --rm -p 8000:8000 deepweather-ai-backend
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/models

WORKDIR /app

# Runtime dependencies. The CPU-only PyTorch index keeps the image far smaller
# than the default CUDA build and is enough for the Phase 0 pipeline.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Application code.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests

# Install the package itself (src layout) so `uvicorn extreme_weather.api.main:app` works.
RUN pip install --no-deps -e . \
    && mkdir -p /app/data/raw /app/data/processed /app/data/samples /app/models

# --- Phase 10: run as a non-root user -------------------------------------
# The service writes only to DATA_DIR and MODEL_DIR, so it needs no root. A
# container escape from a root process is the single highest-impact thing a
# hardening pass can remove, and it costs nothing here.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Liveness only: /health answers "is the process running" and must not depend
# on models or the filesystem, or a missing checkpoint would restart-loop the
# container. Readiness is a separate, stricter check at /ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# No credential is baked in: every setting arrives through the environment.
# Bind to 0.0.0.0 so the port mapping works, which is required inside a
# container and is not by itself an exposure.
CMD ["uvicorn", "extreme_weather.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

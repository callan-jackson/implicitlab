# Container image for free hosting (Render, or anything that runs a container
# and sets $PORT).
#
# The container runs as an unprivileged user that owns the data directory;
# SQLite needs to write next to its file. The filesystem is ephemeral on free
# tiers: uploaded batches and live sessions last until the instance restarts,
# and the demonstration panel is baked into the image, so nothing the demo
# depends on is lost.
#
# BLAS and OpenMP are pinned to one thread. On a fractional-CPU instance, a
# multi-threaded matrix product spends the whole CPU quota in a burst and then
# gets throttled, which makes it slower than doing the same work on one thread.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860 \
    ENVIRONMENT=production \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN useradd -m -u 1000 app
WORKDIR /home/app/implicitlab

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app app ./app
COPY --chown=app:app static ./static
RUN mkdir -p data && chown app:app data

# Bake the 520-person demonstration panel into the image. Simulating it takes a
# second on a real core but over a minute on a free-tier CPU slice, and a cold
# start is exactly when nobody wants to wait.
RUN python -m app.cohort.registry && chown -R app:app cache

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"

# One worker: the free tier has two vCPUs, and the cohort cache is per process,
# so a second worker would double the memory for the demo panel and halve the
# cache hit rate for no throughput a demo needs.
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65 --no-access-log"]

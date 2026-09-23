# Container image for free hosting (Hugging Face Spaces, Docker SDK).
#
# The Space runs the container as uid 1000, so the app directory and the data
# directory are owned by that user; SQLite needs to write next to its file.
# The filesystem is ephemeral: uploaded batches and live sessions last until
# the Space restarts, and the demonstration panel is rebuilt from its seed on
# boot, so nothing the demo depends on is lost.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860 \
    ENVIRONMENT=production

RUN useradd -m -u 1000 app
WORKDIR /home/app/implicitlab

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app app ./app
COPY --chown=app:app static ./static
RUN mkdir -p data && chown app:app data

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"

# One worker: the free tier has two vCPUs, and the cohort cache is per process,
# so a second worker would double the memory for the demo panel and halve the
# cache hit rate for no throughput a demo needs.
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65 --no-access-log"]

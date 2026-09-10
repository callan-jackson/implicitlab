#!/usr/bin/env bash
# Azure App Service (Linux) startup command.
#
# uvicorn is invoked directly rather than through a gunicorn worker class:
# uvicorn's own gunicorn workers are deprecated, and on a single-core plan the
# extra process manager buys nothing. Two workers is enough to keep one
# bootstrap-bound request from blocking a second visitor.
set -e
exec python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 2 \
  --timeout-keep-alive 65 \
  --no-access-log

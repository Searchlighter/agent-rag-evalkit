FROM python:3.11-slim

ARG APP_UID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --uid "${APP_UID}" --shell /usr/sbin/nologin evalkit

COPY pyproject.toml README.md ./
COPY app ./app
COPY reports ./reports
RUN pip install .

USER evalkit

EXPOSE 8000 8001 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

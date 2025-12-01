FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app \
    PORT=8080

WORKDIR $APP_HOME

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared_infrastructure/worker/main.py src/main.py

CMD ["celery", "-A", "src.main", "worker", "--loglevel=info"]


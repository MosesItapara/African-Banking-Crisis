FROM python:3.9-slim

# No .pyc files; print logs immediately instead of buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# XGBoost needs the OpenMP runtime, which the slim image doesn't include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the code, config and raw data
COPY main.py .
COPY config/ config/
COPY src/ src/
COPY data/african_econ_crises.csv data/african_econ_crises.csv

# Don't run as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

CMD ["python", "main.py"]

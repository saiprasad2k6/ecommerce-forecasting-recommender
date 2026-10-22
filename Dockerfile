FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "ecommerce_forecasting_recommender.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.9-slim

WORKDIR /app

COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements-api.txt

COPY app /app/app
COPY src /app/src
COPY config /app/config
COPY models /app/models
COPY examples /app/examples

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

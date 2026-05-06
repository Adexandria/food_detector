FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install torch torchvision && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python || true && \
    pip install --no-cache-dir --prefix=/install opencv-python-headless


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

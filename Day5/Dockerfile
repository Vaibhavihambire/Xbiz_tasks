FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && rm -rf /root/.cache /root/.paddleocr

COPY ocr_engines ./ocr_engines
COPY app.py .
COPY static ./static


RUN mkdir -p transactions

EXPOSE 5500

ENV PYTHONUNBUFFERED=1
ENV DISABLE_MODEL_SOURCE_CHECK=True

CMD ["python", "app.py"]

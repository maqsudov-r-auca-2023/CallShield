FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src CALLSHIELD_ROOT=/app

# serving needs only the inference stack - no torch / lightgbm in the image
RUN pip install --no-cache-dir \
    "fastapi>=0.111" "uvicorn>=0.30" "pydantic>=2.7" "redis>=5.0" \
    "onnxruntime>=1.18" "tokenizers>=0.19" "numpy>=1.26"

COPY src/ src/
COPY models/serving/ models/serving/

EXPOSE 8000
CMD ["uvicorn", "callshield.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

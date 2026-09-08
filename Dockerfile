FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the container needs no network
# access to Hugging Face at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY agent ./agent
COPY api ./api
COPY retrieval ./retrieval
COPY tools ./tools
COPY config.py ./
COPY faiss_index ./faiss_index

# Run as an unprivileged, non-root user. On platforms that assign an arbitrary
# UID at runtime (OpenShift restricted-v2), the process still lands in group 0,
# so make everything under /app group-writable and group-owned by root.
RUN mkdir -p /app/.cache \
    && chgrp -R 0 /app \
    && chmod -R g=u /app
USER 1001

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

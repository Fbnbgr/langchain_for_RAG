FROM python:3.11-slim

WORKDIR /app

# Systemabhängigkeiten für PyPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Abhängigkeiten zuerst (Layer-Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace Modell beim Build cachen (spart Zeit beim Start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Anwendungscode
COPY . .

CMD ["python", "src/ingest.py"]
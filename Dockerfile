FROM python:3.11-slim
WORKDIR /app

# Systemabhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Abhängigkeiten zuerst (Layer-Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# spaCy Modell beim Build laden (nicht zur Laufzeit)
RUN python -m spacy download de_core_news_lg

# HuggingFace Modell beim Build cachen
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Anwendungscode
COPY . .
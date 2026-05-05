
# Langchain RAG Project
## dependencies
ollama
- wird auf localhost:11434 mit Mistral erwartet
- alternativ anderes LLM serven und anpassen
## Schnellstart
### Konfiguration
Alle Konfigurationen erfolgen über die `.env`-Datei.

### development
git clone https://github.com/Fbnbgr/langchain_for_RAG

.env anlegen (siehe evaluation)

pdfs unter data/pdfs ablegen

Beispielfragen und -antworten unter data/evaluation ablegen (siehe unten)

docker compose build

docker compose up -d

docker compose run --rm RAG-pipeline

### deployment
Beispielpdf/Fragen enthalten

docker-compose.yml
```
services:
  RAG-pipeline:
    build: .
    image: fbnbgr/rag:latest
    command: ["python", "src/run_pipeline.py"]  
    working_dir: /app
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
    volumes:
      - ./chroma_db:/app/chroma_db  
      - ./data/evaluation:/app/data/evaluation
    env_file:
      - .env
    extra_hosts:
      - "host.docker.internal:host-gateway"
```   

docker compose pull

.env anlegen (siehe evaluation)

docker compose run --rm RAG-pipeline

## Techstack
Framework
- [LangChain](https://www.langchain.com/)

LLM (Query)
- Mistral 7B Instruct (quantisiert)
- Qwen 2.5 3B (für schwächere Hardware)

Vektordatenbank
- [Chroma DB](https://www.trychroma.com/)

Embedding Model
- deepset/gbert-base per Huggingface

Evaluation
- [RAGAS](https://docs.ragas.io/) für automatisierte RAG-Evaluation

Linting & Code-Qualität
- [Ruff](https://astral.sh/ruff) als schneller Python-Linter und Formatter

## Ablauf
### indexer
- indexiert alle PDFs in data/pdfs und speichert diese in Chroma DB

### retrieval
- Funktionen RAG-Workflow

### evaluation
- evaluiert Beispielfragen aus data/evaluation/examples.py und berechnet RAGAS-Metriken (Faithfulness, Answer Relevancy, Context Relevancy)
- LLM: Mistral
- setup (obligatorisch):
    - import Datei mit Fragen erstellen (data/evaluation/examples.py)
    - .env file muss angelegt werden
    - dataset_name ggf. anpassen

examples.py:

```
examples = [
    {
        "inputs": {"question": "?"},
        "outputs": {"answer": ""},
    },
    {
        "inputs": {"question": "?"},
        "outputs": {"answer": ""},
    },
]
```
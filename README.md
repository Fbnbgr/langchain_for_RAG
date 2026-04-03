
# Langchain RAG Project
## Schnellstart

git clone https://github.com/Fbnbgr/langchain_setup

Download [Mistral 7B from Huggingface](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf) -> Ablage unter ./models

docker compose build

docker compose up -d

docker compose run --rm indexer

docker compose run --rm retrieval

docker compose run --rm evaluation

## Techstack
Framework
- [LangChain](https://www.langchain.com/)

LLM (Query)
- Mistral 7B Instruct (quantisiert)
- Qwen 2.5 3B (für schwächere Hardware)

Vektordatenbank
- [Chroma DB](https://www.trychroma.com/)

Embedding Model
- paraphrase-multilingual-MiniLM-L12-v2 per Huggingface

langsmith
- evaluation

## Ablauf
### indexer
- indexiert alle PDFs in data/pdfs und speichert diese in Chroma DB

### retrieval
- Funktionen RAG.Workflow
- cli bei direktem Aufruf

### evaluation
- evaluiert Beispielfragen aus data/evaluation/examples.py und sendet die Ergebnisse an langsmith
- LLM: Mistral 7B
- setup (obligatorisch):
    - import Datei mit Fragen erstellen (data/evaluation/examples.py)
    - .env file muss angelegt werden

example .env:
```
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```
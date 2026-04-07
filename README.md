
# Langchain RAG Project
## dependencies
ollama
- wird auf localhost:11434 mit Mistral erwartet
- alternativ anderes LLM serven und anpassen
## Schnellstart

git clone https://github.com/Fbnbgr/langchain_setup

.env anlegen (siehe evaluation)

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
- deepset/gbert-base per Huggingface

langsmith
- evaluation

## Ablauf
### indexer
- indexiert alle PDFs in data/pdfs und speichert diese in Chroma DB

### retrieval
- Funktionen RAG-Workflow
- cli bei direktem Aufruf

### evaluation
- evaluiert Beispielfragen aus data/evaluation/examples.py und sendet die Ergebnisse an langsmith
- LLM: Mistral
- setup (obligatorisch):
    - import Datei mit Fragen erstellen (data/evaluation/examples.py)
    - .env file muss angelegt werden
    - dataset_name ggf. anpassen

example .env:
```
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

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

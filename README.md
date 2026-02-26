
# Langchain RAG Project
## Schnellstart
git clone https://github.com/Fbnbgr/langchain_setup
docker compose build
docker compose up -d

docker compose run --rm indexer
docker compose run --rm retrieval

## Techstack
Framework
- [LangChain](https://www.langchain.com/)

LLM (Query)
- Mistral 7B Instruct (quantisiert)
- Qwen 2.5 3B (für schwächere Hardware)

Vektordatenbank
- [Chroma DB](https://www.trychroma.com/)

embedding model
- all-MiniLM-L6-v2 per Huggingface

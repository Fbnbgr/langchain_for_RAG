import logging
import os

import requests
from dotenv import load_dotenv

from evaluation import evaluation
from indexer import run_indexer

logger = logging.getLogger(__name__)
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

def check_ollama() -> None:
    """Prüft ob Ollama erreichbar ist, wirft RuntimeError wenn nicht."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        logger.info(f"Ollama erreichbar: {OLLAMA_BASE_URL}")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Ollama nicht erreichbar unter {OLLAMA_BASE_URL} — läuft der Dienst?")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama antwortet nicht (Timeout) unter {OLLAMA_BASE_URL}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama meldet Fehler: {e}")
    

if __name__ == "__main__":
    check_ollama()
    logger.info("Prozess startet: Indexing")
    run_indexer()
    logger.info("Prozess startet: Evaluation")
    evaluation()
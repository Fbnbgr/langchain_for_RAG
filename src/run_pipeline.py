from venv import logger

from indexer import run_indexer
from evaluation import evaluation

if __name__ == "__main__":
    logger.info(f"Prozess startet: Indexing")
    run_indexer()
    logger.info(f"Prozess startet: Evaluation")
    evaluation()